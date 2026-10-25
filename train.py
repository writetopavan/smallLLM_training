"""Offline training smoke test; CUDA required by default."""
import argparse
import json
import math
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    try:
        import torch
    except ImportError:
        raise SystemExit('PyTorch missing. Run on your existing RunPod PyTorch environment.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('FAIL: CUDA unavailable. Use the pod or explicitly pass --device cpu.')
    torch.manual_seed(42)
    if args.device == 'cpu':
        torch.set_num_threads(2)
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(0) if device.type == 'cuda' else None
    print(f'PyTorch: {torch.__version__} | Device: {gpu or device}', flush=True)
    x = torch.randn(2048, 16, device=device)
    y = x @ torch.randn(16, 1, device=device)
    model = torch.nn.Linear(16, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.MSELoss()
    initial = loss_fn(model(x), y).item()
    started = time.perf_counter()
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 50 == 0:
            print(f'Step {step:3d} | loss: {loss.item():.6f}', flush=True)
    with torch.no_grad():
        final = loss_fn(model(x), y).item()
    elapsed = time.perf_counter() - started
    output = Path(__file__).resolve().parent / 'outputs'
    output.mkdir(exist_ok=True)
    model.cpu()
    torch.save(model.state_dict(), output / 'model.pt')
    restored = torch.nn.Linear(16, 1)
    restored.load_state_dict(torch.load(output / 'model.pt', map_location='cpu', weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(x[:8].cpu()), model(x[:8].cpu()))
    passed = math.isfinite(final) and final < initial * 0.1
    metrics = dict(passed=passed, device=str(device), gpu=gpu, initial_loss=initial,
                   final_loss=final, seconds=elapsed, checkpoint_reload_passed=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    if not passed:
        raise SystemExit('FAIL: loss did not decrease sufficiently.')
    print('PASS: Training, loss reduction, checkpoint save and reload verified.')


if __name__ == '__main__':
    main()
