# Local to RunPod test

Tiny offline PyTorch training project. No dataset downloads or tokens needed.
Use your existing RunPod PyTorch environment; no dependency install is needed there.

In Windows PowerShell, open this folder and replace the IP and port with the
RunPod direct SSH TCP details (the proxy endpoint may not support SCP):

```powershell
$podAddress = 'YOUR_POD_IP'
$podPort = 12345 # Replace with actual port
./pod.ps1 upload -PodHost $podAddress -Port $podPort
./pod.ps1 run -PodHost $podAddress -Port $podPort
./pod.ps1 download -PodHost $podAddress -Port $podPort
```

If needed add `-KeyPath 'C:\path\to\key'` to each command.
Upload copies source only, preserving remote outputs. Repeat it after edits.
Run is a short foreground job: wait for completion before downloading.
Results download to `pod-results/outputs`.

For VS Code Remote SSH, open `/workspace/ML_training` on the pod and run:

```bash
python -u train.py
```

Expect the GPU name, decreasing loss and PASS. Outputs: `outputs/model.pt`
and `outputs/metrics.json`. The script checks checkpoint save/reload and loss
reduction. CUDA is required by default to catch accidental local execution.

Optional local test, with local PyTorch installed:

```powershell
python train.py --device cpu
```

For long jobs use tmux on the pod (`tmux new -s training`), then run training.
Detach with Ctrl+B then D; reconnect with `tmux attach -t training`.
Install tmux first if unavailable. Verify storage persistence before terminating
a pod and download important checkpoints. This project tests infrastructure,
not LLM fine-tuning performance.

## Git workflow

On the pod, clone once (this repository is public):

```bash
cd /workspace
git clone https://github.com/writetopavan/smallLLM_training.git
cd smallLLM_training
python -u train.py
```

After committing and pushing changes from your laptop, update the pod:

```bash
cd /workspace/smallLLM_training
git pull --ff-only
python -u train.py
```

Keep edits on your laptop when using this workflow. Generated checkpoints,
results, and logs are ignored by Git. The PowerShell SCP workflow above remains
an alternative and uses the separate `/workspace/ML_training` directory.
