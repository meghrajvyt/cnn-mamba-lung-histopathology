"""VMamba, vendored from its authors (Liu et al., NeurIPS 2024).

Source  https://github.com/MzeroMiko/VMamba, file ``vmamba.py`` (the authors'
        single-file implementation), commit
        2ed52ead062a51a64521ed3871d52914bf532876 (2025-03-07).
Licence MIT, Copyright (c) 2024 MzeroMiko; the licence text is in LICENSE here.

Changed: imports only. ``from fvcore.nn import ...`` is wrapped in
try/except, because fvcore is used only by ``VSSM.flops()``, which this study
does not call. Nothing else differs from the upstream file
(SHA-256 572207a2b6553ff85d81b602eebda2041623073f01fc0c6d1280c5a2add040f6);
tests/test_vmamba.py pins the vendored file's own hash so it cannot drift.

The selective scan runs on mamba-ssm's CUDA kernel (``selective_scan_cuda``)
when it is installed, and otherwise on the authors' pure-PyTorch reference
(``selective_scan_torch``), which is exact but slow. tools/check_vmamba_kernel.py
compares the two before any training run (docs/added_analyses_protocol.md).
"""
