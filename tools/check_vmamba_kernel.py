#!/usr/bin/env python3
"""Check VMamba's fused kernels against the authors' PyTorch reference.

Run on the GPU before any VMamba-T training (docs/added_analyses_protocol.md, A1).

  1. Selective scan. For the four stage shapes of VMamba-T at 224 px, random
     inputs go through mamba-ssm's CUDA kernel (``SelectiveScanCuda``) and the
     authors' pure-PyTorch reference (``selective_scan_torch``). The outputs and
     the gradients of all seven inputs are compared, in fp32 and with fp16
     activations, as under the mixed precision used for training.
  2. Cross-scan and cross-merge. The Triton kernels against the PyTorch ones,
     outputs and gradients.
  3. Whole model. VMamba-T with the authors' ImageNet-1K weights (hash-checked,
     loaded strictly) runs on random images in IEEE fp32 (TF32 off) with the
     fused kernels and with the references; logits and the gradient with
     respect to the input are compared.
  4. Smoke test. Mixed-precision training steps at batch 32: finite loss, time
     per step and peak memory.

The maximum absolute and relative differences, versions and hashes are written
to --out. The exit status is non-zero if any check fails.

Usage:
    python3 tools/check_vmamba_kernel.py --out logs/vmamba_kernel_check.txt
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ModelConfig  # noqa: E402
from models import baselines  # noqa: E402

# Relative tolerances: max|kernel - reference| / max|reference|. They are loose
# on purpose; a kernel built wrongly is off by O(1), not by rounding.
TOL = {"fp32": 1e-3, "fp16": 1e-2}
STAGES = ((96, 56), (192, 28), (384, 14), (768, 7))  # (d_inner, side) at 224 px
K, N = 4, 1  # scan directions; d_state of vssm1_tiny_0230s


def rel(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    a, b = a.detach().float(), b.detach().float()
    err = (a - b).abs().max().item()
    return err, err / max(b.abs().max().item(), 1e-30)


def scan_inputs(d_inner: int, length: int, dtype: torch.dtype, device) -> List[torch.Tensor]:
    """Random inputs in the ranges the model produces (mamba's dt initialisation)."""
    kd, bsz = K * d_inner, 2
    dt = torch.exp(torch.rand(kd, device=device) * (math.log(0.1) - math.log(1e-3))
                   + math.log(1e-3))
    delta_bias = dt + torch.log(-torch.expm1(-dt))       # inverse softplus
    tensors = [
        torch.randn(bsz, kd, length, device=device).to(dtype),           # u
        (0.5 * torch.randn(bsz, kd, length, device=device)).to(dtype),   # delta
        -torch.exp(0.5 * torch.randn(kd, N, device=device)),             # A < 0
        torch.randn(bsz, K, N, length, device=device).to(dtype),         # B
        torch.randn(bsz, K, N, length, device=device).to(dtype),         # C
        torch.randn(kd, device=device),                                  # D
        delta_bias,
    ]
    return [t.requires_grad_(True) for t in tensors]


def run_scan(fn, inputs, grad_out):
    leaves = [t.detach().clone().requires_grad_(True) for t in inputs]
    out = fn(*leaves)
    (out.float() * grad_out).sum().backward()
    return out, [t.grad for t in leaves]


def check_scan(vm, lines: List[str]) -> bool:
    ok = True
    names = ("u", "delta", "A", "B", "C", "D", "delta_bias")
    kernel = lambda *t: vm.SelectiveScanCuda.apply(*t, True, True, None)  # noqa: E731
    ref = lambda *t: vm.selective_scan_torch(*t, True, True)  # noqa: E731
    for prec, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
        for d_inner, side in STAGES:
            torch.manual_seed(d_inner)
            inputs = scan_inputs(d_inner, side * side, dtype, "cuda")
            grad_out = torch.randn(2, K * d_inner, side * side, device="cuda")
            out_k, g_k = run_scan(kernel, inputs, grad_out)
            out_r, g_r = run_scan(ref, inputs, grad_out)
            row = [("out", *rel(out_k, out_r))]
            row += [(f"d{n}", *rel(a, b)) for n, a, b in zip(names, g_k, g_r)]
            worst = max(r for _, _, r in row)
            good = worst < TOL[prec] and all(math.isfinite(e) for _, e, _ in row)
            ok &= good
            lines.append(
                f"scan {prec} d_inner={d_inner} L={side * side}: "
                + " ".join(f"{n} {e:.2e} ({r:.1e})" for n, e, r in row)
                + f" -> {'PASS' if good else 'FAIL'}")
    return ok


def check_cross_scan(vm, lines: List[str]) -> bool:
    ok = True
    for c, side in STAGES:
        torch.manual_seed(c)
        x = torch.randn(2, c, side, side, device="cuda")
        y = torch.randn(2, K, c, side, side, device="cuda")
        for name, fn, inp in (("cross_scan", vm.cross_scan_fn, x),
                              ("cross_merge", vm.cross_merge_fn, y)):
            res = []
            for force_torch in (False, True):
                leaf = inp.detach().clone().requires_grad_(True)
                out = fn(leaf, in_channel_first=True, out_channel_first=True,
                         force_torch=force_torch)
                torch.manual_seed(1)  # the same upstream gradient for both
                g = torch.randn(out.shape, device="cuda")
                (out * g).sum().backward()
                res.append((out, leaf.grad))
            (eo, ro), (eg, rg) = rel(res[0][0], res[1][0]), rel(res[0][1], res[1][1])
            good = max(ro, rg) < TOL["fp32"]
            ok &= good
            lines.append(f"{name} (Triton vs PyTorch) C={c} {side}x{side}: out {eo:.2e} "
                         f"({ro:.1e}) grad {eg:.2e} ({rg:.1e}) -> {'PASS' if good else 'FAIL'}")
    return ok


def model_outputs(model, x, w):
    leaf = x.detach().clone().requires_grad_(True)
    logits = model(leaf)
    (logits * w).sum().backward()
    return logits.detach(), leaf.grad.detach()


def check_model(vm, lines: List[str]) -> Tuple[bool, torch.nn.Module]:
    cfg = ModelConfig()
    cfg.num_classes = 3
    cfg.image_size = 224
    torch.manual_seed(0)
    model = baselines.build_model("vmamba_tiny", cfg).cuda().eval()  # strict load
    lines.append(f"pretrained weights: {baselines.vmamba_t_weights_path()} sha256 "
                 f"{baselines.sha256_file(baselines.vmamba_t_weights_path())} "
                 f"(expected {baselines.VMAMBA_T_SHA256}); loaded with strict=True")
    lines.append(f"parameters (3-class head): {model.count_parameters()['total']:,}")
    torch.manual_seed(1)
    x = torch.randn(2, 3, 224, 224, device="cuda")
    w = torch.randn(2, 3, device="cuda")
    # IEEE fp32 throughout, so that only the kernels differ: with cuDNN's default
    # TF32 convolutions, two runs of the same fused model already differ by
    # about 1e-3 (relative) in the input gradient.
    tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    saved = vm.WITH_SELECTIVESCAN_MAMBA, vm.WITH_TRITON
    try:
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        logits_k, gx_k = model_outputs(model, x, w)
        # both module flags are read at call time
        vm.WITH_SELECTIVESCAN_MAMBA, vm.WITH_TRITON = False, False
        logits_r, gx_r = model_outputs(model, x, w)
    finally:
        vm.WITH_SELECTIVESCAN_MAMBA, vm.WITH_TRITON = saved
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32
    (el, rl), (eg, rg) = rel(logits_k, logits_r), rel(gx_k, gx_r)
    good = max(rl, rg) < TOL["fp32"]
    lines.append(f"model fp32, TF32 off (fused vs reference), 2 images 224 px: logits {el:.2e} "
                 f"({rl:.1e}) d/dinput {eg:.2e} ({rg:.1e}) -> {'PASS' if good else 'FAIL'}")
    return good, model


def smoke_test(model, lines: List[str], batch: int) -> bool:
    model.train()
    opt = torch.optim.AdamW(model.get_parameter_groups(lr=1e-3, weight_decay=0.05))
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.CrossEntropyLoss()
    torch.manual_seed(2)
    x = torch.randn(batch, 3, 224, 224, device="cuda")
    y = torch.randint(0, 3, (batch,), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    losses, times = [], []
    for _ in range(4):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = lossf(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(loss.item())
    good = all(math.isfinite(v) for v in losses)
    lines.append(f"smoke test, AMP fp16 training at batch {batch}: losses "
                 f"{' '.join(f'{v:.4f}' for v in losses)}; {min(times[1:]):.3f} s/step "
                 f"after warm-up; peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"
                 f" -> {'PASS' if good else 'FAIL'}")
    return good


def versions(vm) -> List[str]:
    def ver(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    kernel = sys.modules.get("selective_scan_cuda")
    return [
        f"date: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        f"gpu: {torch.cuda.get_device_name(0)}; torch {torch.__version__}; "
        f"CUDA {torch.version.cuda}; mamba_ssm {ver('mamba_ssm')}; triton {ver('triton')}; "
        f"timm {ver('timm')}",
        f"selective_scan_cuda: {getattr(kernel, '__file__', 'NOT IMPORTED')}",
        f"vendored vmamba.py sha256 {baselines.sha256_file(Path(vm.__file__))}",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "logs/vmamba_kernel_check.txt"))
    ap.add_argument("--smoke_batch", type=int, default=32,
                    help="batch of the smoke test (32, the training batch, by default)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a CUDA GPU")
        return 2
    from models.vmamba_vendor import vmamba as vm

    lines = ["VMamba-T kernel check (tools/check_vmamba_kernel.py); "
             "differences are max |fused - reference| (relative to max |reference|)"]
    lines += versions(vm)
    if not (vm.WITH_SELECTIVESCAN_MAMBA and vm.WITH_TRITON):
        lines.append(f"FAIL: selective_scan_cuda available={vm.WITH_SELECTIVESCAN_MAMBA}, "
                     f"triton available={vm.WITH_TRITON}")
        ok = False
    else:
        ok = check_scan(vm, lines)
        ok &= check_cross_scan(vm, lines)
        good, model = check_model(vm, lines)
        ok &= good
        ok &= smoke_test(model, lines, args.smoke_batch)
    lines.append(f"RESULT: {'PASS' if ok else 'FAIL'} (tolerances {TOL})")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
