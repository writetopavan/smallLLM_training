"""LoRA practice: extract payment terms from synthetic contract clauses."""
import argparse
import gc
import json
import math
import random
from pathlib import Path

SYSTEM = ('Extract payment terms from the clause. Return only JSON with keys '
          'payment_days (integer) and late_fee_percent (number or null).')


def examples():
    rows = []
    for days in (7, 10, 15, 20, 30, 45, 60, 90):
        for fee in (None, 0.5, 1, 1.5, 2):
            for wording in range(3):
                clauses = [f'Invoices are payable within {days} days of receipt.',
                           f'The customer shall pay each invoice no later than {days} days after receipt.',
                           f'Payment is due {days} days from receipt of the invoice.']
                clause = clauses[wording]
                if fee is not None:
                    clause += f' Overdue amounts incur a late fee of {fee}% per month.'
                answer = json.dumps(dict(payment_days=days, late_fee_percent=fee))
                rows.append(dict(clause=clause, answer=answer))
    random.Random(42).shuffle(rows)
    return rows[:100], rows[100:]


def encode(row, tokenizer):
    # Separate tokenization gives an explicit boundary for response-only loss.
    prefix = tokenizer.apply_chat_template([
        dict(role='system', content=SYSTEM), dict(role='user', content=row['clause'])
    ], tokenize=True, add_generation_prompt=True)
    completion = tokenizer(row['answer'], add_special_tokens=False)['input_ids']
    completion += [tokenizer.eos_token_id]
    ids = prefix + completion
    if len(ids) > 256:
        raise ValueError('Example exceeds 256 tokens; do not silently truncate the answer.')
    return dict(input_ids=ids, attention_mask=[1] * len(ids),
                labels=[-100] * len(prefix) + completion)


def generate(model, tokenizer, clause):
    import torch
    text = tokenizer.apply_chat_template([
        dict(role='system', content=SYSTEM), dict(role='user', content=clause)
    ], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    model.eval()
    with torch.no_grad():
        tokens = model.generate(**inputs, max_new_tokens=80, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True)
    return tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/lora-qwen'))
    parser.add_argument('--resume', action='store_true', help='Resume latest checkpoint in this output directory')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required. Run this script on your GPU pod.')
    checkpoint = get_last_checkpoint(str(args.output_dir)) if args.output_dir.exists() else None
    if args.resume and checkpoint is None:
        raise SystemExit('No checkpoint found to resume.')
    if not args.resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Output directory is not empty. Use --resume or a different --output-dir.')
    set_seed(42)
    print(f'GPU: {torch.cuda.get_device_name(0)} | model: {args.model}', flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                attn_implementation='sdpa').to('cuda')
    probe = 'Invoices must be paid within 25 days of receipt. A late fee of 1.5% per month applies.'
    before = generate(model, tokenizer, probe)
    print('BASE MODEL:', before, flush=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05, bias='none',
        target_modules=['q_proj', 'v_proj']))
    model.print_trainable_parameters()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in name for name in trainable):
        raise RuntimeError('Expected only LoRA weights to be trainable.')
    train_rows, eval_rows = examples()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'dataset.json').write_text(json.dumps(
        dict(train=train_rows, evaluation=eval_rows), indent=2), encoding='utf-8')
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), max_steps=args.steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        per_device_eval_batch_size=2, learning_rate=2e-4,
        warmup_steps=min(5, args.steps // 5), bf16=dtype == torch.bfloat16,
        logging_steps=5, logging_first_step=True, eval_strategy='steps', eval_steps=20,
        save_strategy='steps', save_steps=20, save_total_limit=2,
        report_to='none', optim='adamw_torch', seed=42,
        dataloader_num_workers=0, label_names=['labels'])
    trainer = Trainer(model=model, args=training_args,
        train_dataset=[encode(row, tokenizer) for row in train_rows],
        eval_dataset=[encode(row, tokenizer) for row in eval_rows],
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100,
                                           pad_to_multiple_of=8),
        processing_class=tokenizer)
    initial = trainer.evaluate()['eval_loss']
    trained = trainer.train(resume_from_checkpoint=checkpoint if args.resume else None)
    final = trainer.evaluate()['eval_loss']
    if not math.isfinite(final):
        raise RuntimeError('Non-finite evaluation loss.')
    adapter_dir = args.output_dir / 'adapter'
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    after = generate(model, tokenizer, probe)
    # Verify a fresh base model can load the saved adapter and reproduce generation.
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                               attn_implementation='sdpa').to('cuda')
    reloaded = PeftModel.from_pretrained(base, adapter_dir)
    restored = generate(reloaded, tokenizer, probe)
    metrics = dict(model=args.model, steps=args.steps, initial_eval_loss=initial,
        final_eval_loss=final, eval_loss_improved=final < initial,
        train_metrics=trained.metrics, probe=probe, before=before, after=after,
        reloaded_output=restored, reload_matches=restored == after,
        peak_gpu_memory_gb=torch.cuda.max_memory_allocated() / 1024**3)
    (args.output_dir / 'results.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    if restored != after:
        raise RuntimeError('Reloaded adapter produced different deterministic output; inspect results.json.')
    print(f'COMPLETE: adapter saved to {adapter_dir}', flush=True)


if __name__ == '__main__':
    main()
