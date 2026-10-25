"""Compare base versus saved LoRA using generation; does not train anything."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

from train_lora import generate


def parse_response(text):
    """Report strict JSON compliance separately from fenced-JSON content."""
    candidate = text.strip()
    strict = True
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', candidate, re.DOTALL)
    if fence:
        strict = False
        candidate = fence.group(1)
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return False, None
    if not isinstance(parsed, dict):
        return False, None
    return strict, parsed


def score(text, expected):
    strict, parsed = parse_response(text)
    valid_days = parsed is not None and type(parsed.get('payment_days')) is int
    fee = parsed.get('late_fee_percent') if parsed is not None else None
    valid_fee = parsed is not None and 'late_fee_percent' in parsed and (
        fee is None or (type(fee) in (int, float) and math.isfinite(fee)))
    schema = (valid_days and valid_fee and set(parsed) == set(expected))
    days_ok = valid_days and parsed['payment_days'] == expected['payment_days']
    fee_ok = valid_fee and (fee == expected['late_fee_percent'] if fee is None or
        expected['late_fee_percent'] is None else math.isclose(
            fee, expected['late_fee_percent'], rel_tol=0, abs_tol=1e-8))
    return dict(strict_json=strict, schema_valid=bool(schema),
        payment_days_correct=bool(days_ok), late_fee_correct=bool(fee_ok),
        both_fields_correct=bool(days_ok and fee_ok),
        strict_exact_match=bool(strict and schema and days_ok and fee_ok))


def challenge_cases():
    # Explicit payment deadlines only; percentage is copied, never annualized.
    cases = [
        ('unseen_values', 'Pay invoices within 25 days of receipt. The monthly late fee is 1.5%.', 25, 1.5),
        ('unseen_values', 'Payment is due 75 days after invoice receipt. Overdue amounts incur 0.75% per month.', 75, 0.75),
        ('words', 'Invoices must be settled within thirty days of receipt. The late fee is one percent per month.', 30, 1),
        ('paraphrase', 'The buyer has 45 calendar days after receiving an invoice to remit payment. Late balances attract a monthly charge of 2%.', 45, 2),
        ('missing_fee', 'Settle each invoice within 30 days of receipt.', 30, None),
        ('explicit_zero', 'Payment is due within 15 days. The late payment fee is 0%.', 15, 0),
        ('distractor', 'VAT is 18%. Invoices are payable within 20 days. Overdue balances incur a 1.5% late fee per month.', 20, 1.5),
        ('distractor', 'A 2% early-payment discount is available. The final payment deadline is 60 days after receipt. No late fee is specified.', 60, None),
        ('reordered', 'A late fee of 0.5% per month applies to overdue balances. Invoices must be paid within 45 days.', 45, 0.5),
        ('negation', 'No late payment fee shall be charged. All invoices are payable within 10 days.', 10, None),
        ('distractor', 'The agreement lasts 365 days. Invoices must be paid within 35 days. Late payments carry a 1.25% monthly fee.', 35, 1.25),
        ('format', 'PAYMENT DEADLINE: 30 days after invoice receipt. LATE PAYMENT FEE: 2.25% monthly.', 30, 2.25),
    ]
    return [dict(id=f'challenge-{i+1}', suite='challenge', category=category,
                 clause=clause, expected=dict(payment_days=days, late_fee_percent=fee))
            for i, (category, clause, days, fee) in enumerate(cases)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, default=Path('outputs/lora-qwen/adapter'))
    parser.add_argument('--dataset', type=Path, help='Defaults to dataset.json alongside adapter')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/lora-evaluation'))
    parser.add_argument('--clause', help='Optional custom clause; printed but not scored')
    args = parser.parse_args()
    config_path = args.adapter / 'adapter_config.json'
    if not config_path.is_file():
        parser.error(f'Adapter not found: {config_path}')
    dataset_path = args.dataset or args.adapter.parent / 'dataset.json'
    if not dataset_path.is_file():
        parser.error(f'Training dataset snapshot not found: {dataset_path}')
    snapshot = json.loads(dataset_path.read_text(encoding='utf-8'))
    cases = [dict(id=f'heldout-{i+1}', suite='heldout', category='training_distribution',
                  clause=row['clause'], expected=json.loads(row['answer']))
             for i, row in enumerate(snapshot['evaluation'])]
    if not cases:
        parser.error('Evaluation split is empty')
    cases += challenge_cases()
    # Missing deadlines and competing conditional terms have no defined label in
    # the training schema; inspect them qualitatively instead of inventing gold.
    exploratory = [
        'Overdue amounts incur a fee of 1.5% per month. No payment deadline is stated.',
        'Payment is due within 15 days for services and 60 days for equipment. Late fees are 1% per month.',
        'Invoices must be paid promptly. Late fees will be negotiated separately.',
    ]
    if args.clause:
        exploratory.append(args.clause)
    cases += [dict(id=f'exploratory-{i+1}', suite='exploratory', category='unscored',
                   clause=clause, expected=None) for i, clause in enumerate(exploratory)]
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required. Run inference on the pod.')
    set_seed(42)
    model_id = json.loads(config_path.read_text())['base_model_name_or_path']
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                               attn_implementation='sdpa').to('cuda')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    # Capture true base outputs before PEFT attaches adapters to the model.
    for index, case in enumerate(cases, 1):
        print(f'BASE {index}/{len(cases)} {case["id"]}', flush=True)
        results.append(dict(**case, base_output=generate(base, tokenizer, case['clause'])))
    model = PeftModel.from_pretrained(base, args.adapter)
    for index, row in enumerate(results, 1):
        print(f'LORA {index}/{len(results)} {row["id"]}', flush=True)
        row['lora_output'] = generate(model, tokenizer, row['clause'])
        if row['expected'] is not None:
            row['base_score'] = score(row['base_output'], row['expected'])
            row['lora_score'] = score(row['lora_output'], row['expected'])
    summary = {}
    for suite in ('heldout', 'challenge'):
        subset = [r for r in results if r['suite'] == suite]
        summary[suite] = dict(count=len(subset))
        for variant in ('base', 'lora'):
            keys = subset[0][f'{variant}_score']
            summary[suite][variant] = {key: sum(r[f'{variant}_score'][key] for r in subset)
                                       / len(subset) for key in keys}
    payload = dict(model=model_id, adapter=str(args.adapter), dataset=str(dataset_path),
                   prompt='Same system prompt and greedy generation as training probe',
                   summary=summary, examples=results)
    (args.output_dir / 'evaluation.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with (args.output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['id', 'suite', 'category', 'clause',
                                                   'expected', 'base_output', 'lora_output'])
        writer.writeheader()
        for row in results:
            writer.writerow({key: json.dumps(row[key]) if key == 'expected' else row[key]
                             for key in writer.fieldnames})
    print('\nAccuracy fractions (1.0 = 100%):', flush=True)
    print(json.dumps(summary, indent=2))
    for row in results:
        if row['expected'] is None or not row['lora_score']['strict_exact_match']:
            print(f'\n{row["id"]}: {row["clause"]}\nExpected: {row["expected"]}'
                  f'\nBase: {row["base_output"]}\nLoRA: {row["lora_output"]}')
    print(f'\nSaved detailed reports to {args.output_dir}')


if __name__ == '__main__':
    main()
