"""Phase 1 verification: parallel selective scan == reference sequential scan.

Run:  python3 _verify_scan.py
"""
import time

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from models.ssm_block import SelectiveSSM

torch.manual_seed(0)


def make_inputs(batch, seqlen, ssm, device, scale=1.0):
    """Build (u, delta, A, B, C) with realistic shapes/ranges."""
    d_inner, d_state = ssm.d_inner, ssm.d_state
    u = torch.randn(batch, seqlen, d_inner, device=device)
    # delta comes from softplus in the real forward pass, so strictly positive
    delta = torch.nn.functional.softplus(
        torch.randn(batch, seqlen, d_inner, device=device) * scale
    )
    A = -torch.exp(ssm.A_log.detach().to(device))  # (d_inner, d_state), < 0
    B = torch.randn(batch, seqlen, d_state, device=device)
    C = torch.randn(batch, seqlen, d_state, device=device)
    return u, delta, A, B, C


def check_equivalence(device):
    print(f"\n=== Numerical equivalence on {device} ===")
    ssm = SelectiveSSM(d_model=192, d_state=16).to(device).eval()
    ok = True

    for seqlen in (1, 7, 16, 32, 33, 64, 196, 197):
        for chunk in (1, 8, 32, 64, 0):
            ssm.scan_chunk_size = chunk
            u, delta, A, B, C = make_inputs(2, seqlen, ssm, device)
            with torch.no_grad():
                fast = ssm._selective_scan(u, delta, A, B, C)
                ref = ssm._selective_scan_reference(u, delta, A, B, C)
            err = (fast - ref).abs().max().item()
            denom = ref.abs().max().item() or 1.0
            rel = err / denom
            good = rel < 1e-5
            ok &= good
            if not good:
                print(f"  FAIL L={seqlen:4d} chunk={chunk:3d} rel_err={rel:.2e}")
    print("  all shapes/chunks match" if ok else "  MISMATCH FOUND")

    # Stress test: large delta => large |delta*A|, where the naive cumsum
    # formulation would overflow. The associative form must stay finite.
    ssm.scan_chunk_size = 32
    u, delta, A, B, C = make_inputs(2, 196, ssm, device, scale=6.0)
    with torch.no_grad():
        fast = ssm._selective_scan(u, delta, A, B, C)
        ref = ssm._selective_scan_reference(u, delta, A, B, C)
    print(f"  stress (max|dA_log|={(delta.unsqueeze(-1)*A).abs().max():.1f}): "
          f"finite={torch.isfinite(fast).all().item()} "
          f"rel_err={((fast-ref).abs().max()/ref.abs().max()).item():.2e}")

    # Gradients must match too, not just forward values.
    ssm.scan_chunk_size = 32
    u, delta, A, B, C = make_inputs(2, 64, ssm, device)
    grads = {}
    for name, fn in (("fast", ssm._selective_scan),
                     ("ref", ssm._selective_scan_reference)):
        uu = u.clone().requires_grad_(True)
        dd = delta.clone().requires_grad_(True)
        fn(uu, dd, A, B, C).sum().backward()
        grads[name] = (uu.grad.clone(), dd.grad.clone())
    gu = (grads["fast"][0] - grads["ref"][0]).abs().max().item()
    gd = (grads["fast"][1] - grads["ref"][1]).abs().max().item()
    print(f"  grad match: max|d/du| diff={gu:.2e}  max|d/ddelta| diff={gd:.2e}")
    return ok


def benchmark(device):
    print(f"\n=== Speed on {device} (batch=4, L=196, d_model=192) ===")
    ssm = SelectiveSSM(d_model=192, d_state=16).to(device)
    u, delta, A, B, C = make_inputs(4, 196, ssm, device)

    u = u.requires_grad_(True)
    delta = delta.requires_grad_(True)

    def timed(fn, iters=3):
        # warmup
        out = fn(u, delta, A, B, C)
        out.sum().backward()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(u, delta, A, B, C)
            out.sum().backward()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    t_ref = timed(ssm._selective_scan_reference)
    row = []
    for chunk in (8, 32, 64, 196):
        ssm.scan_chunk_size = chunk
        row.append((chunk, timed(ssm._selective_scan)))

    print(f"  reference loop (196 steps) : {t_ref * 1e3:8.1f} ms")
    for chunk, t in row:
        steps = -(-196 // chunk) * max(1, (chunk - 1).bit_length())
        print(f"  parallel chunk={chunk:3d} (~{steps:3d} steps): "
              f"{t * 1e3:8.1f} ms   speedup {t_ref / t:5.1f}x")


if __name__ == "__main__":
    all_ok = check_equivalence("cpu")
    benchmark("cpu")
    if torch.backends.mps.is_available():
        benchmark("mps")
    print("\nRESULT:", "PASS" if all_ok else "FAIL")
