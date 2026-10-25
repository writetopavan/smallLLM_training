# LoRA fine-tuning practice

Model: Qwen/Qwen2.5-0.5B-Instruct. Standard LoRA (no quantization): rank 8,
alpha 16, query/value projections. The base model is frozen. Only adapter weights
are trained, using response-only cross-entropy (prompt/padding labels are -100).

## Run on your pod

Use /root for Git because your /workspace volume rejects Git chmod operations.
Your existing outputs symlink sends models/results to /workspace/training-outputs.

```bash
cd /root/smallLLM_training
git pull --ff-only
# Isolate dependencies while reusing the pod's existing CUDA-enabled PyTorch.
python -m venv --system-site-packages /root/lora-env
source /root/lora-env/bin/activate
python -m pip install -r requirements-lora.txt
python -m pip check
python -u train_lora.py
```

If pip check reports conflicts from unrelated image packages, retain the output
for diagnosis. Do not replace the working CUDA torch install blindly.
The initial run downloads public model weights/tokenizer from Hugging Face
(roughly 1 GB for weights); network access and free disk are required, no token
normally needed. A40 has ample memory for these small defaults. Runtime depends
on download speed and pod load; no fixed runtime is promised.

Data: 120 deterministic synthetic payment clauses, 100 train / 20 held out.
The task extracts payment_days and late_fee_percent into JSON. Data is generated
locally and saved with results. Held-out examples share templates and values
with training: evaluation loss measures this toy task, not real-world accuracy.
A separate probe uses an unseen 25-day payment term. Base and fine-tuned outputs
are recorded, not assumed to improve. This is not legal advice or production data.

## Output

- outputs/lora-qwen/adapter: LoRA adapter and tokenizer (requires base model).
- outputs/lora-qwen/results.json: before/after generations, held-out losses,
  memory, training metrics and adapter reload check.
- outputs/lora-qwen/checkpoint-*: resumable training state every 20 steps.

The script reloads the adapter onto a fresh base and compares deterministic
output. Loss improvement is reported separately, not treated as proof of quality.

Short plumbing check (may not improve outputs):

```bash
python -u train_lora.py --steps 5 --output-dir outputs/lora-smoke
```

Resume an interrupted default run from its latest checkpoint:

```bash
python -u train_lora.py --resume
```

For a separate experiment, use a new output directory. Existing nonempty runs
are protected from accidental overwrite. The script is CUDA-only.
For longer runs, start tmux before training so SSH disconnects do not stop it.
Do not push outputs, checkpoints, caches or tokens to GitHub.

## Sources

- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- https://huggingface.co/docs/peft/guides/peft_model_config
- https://huggingface.co/docs/transformers/v4.56.2/main_classes/trainer

## Post-training inference evaluation

```bash
cd /root/smallLLM_training
git pull --ff-only
source /root/lora-env/bin/activate
python -u evaluate_lora.py
```

No training occurs. Loads the saved adapter and its recorded base model.
Scores the actual 20 held-out rows from the saved dataset.json (not a regenerated
split) plus 12 hand-labeled challenge clauses. Another 3 ambiguous/missing-deadline
clauses are qualitative, unscored probes: the training schema cannot represent
all of them faithfully. This is a tiny diagnostic suite, not a production benchmark.

Reports separate heldout/challenge fractions for strict JSON, schema validity,
payment-days accuracy, late-fee accuracy, both-fields accuracy, and strict exact
match. Field accuracy accepts a single Markdown JSON fence; strict exact match
requires a bare JSON object with exactly the expected keys and correct types.
Absent or explicitly waived late fees use null; an explicitly stated 0% uses 0.
Percentages are copied as stated, not converted to fractions or annualized.

Results: outputs/lora-evaluation/evaluation.json and predictions.csv. These use
your existing outputs symlink. Re-running evaluation replaces these two reports.
No model weights or training checkpoints are modified.

Optional custom inference, appended as an unscored probe:

```bash
python -u evaluate_lora.py --clause 'Payment is due in 25 days. Late fees are 0.75% per month.'
```
