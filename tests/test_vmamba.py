"""VMamba-T comparator (models/vmamba_vendor, models/baselines.VMambaClassifier).

Runs on the CPU with the authors' pure-PyTorch selective scan, at small inputs.
The CUDA kernel is checked against that reference on the GPU by
tools/check_vmamba_kernel.py, before any training run.
"""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from config import ModelConfig
from models import baselines

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "models" / "vmamba_vendor" / "vmamba.py"
# Upstream file (commit 2ed52ea) with only the fvcore import made optional.
VENDORED_SHA256 = "7fb6b83cb5e047669eee57f4214dd9a09192992f9711ae33aa4eb3221bb6b67e"
N_PARAMS_3_CLASSES = 29_482_371


@pytest.fixture(scope="module")
def vm():
    """The vendored module. It applies @triton.jit at import time, so where Triton
    is not installed (macOS) a stub stands in while it is imported; Triton is
    only called for CUDA tensors, so the CPU path never reaches the stub."""
    if importlib.util.find_spec("triton") is not None:
        from models.vmamba_vendor import vmamba
        return vmamba
    import timm.layers  # noqa: F401  resolve Triton's absence before the stub exists
    import timm.models.layers  # noqa: F401
    import torch._dynamo  # noqa: F401
    stub = types.ModuleType("triton")
    stub.jit = lambda fn: fn
    stub.language = types.ModuleType("triton.language")
    stub.language.tensor = stub.language.constexpr = object
    sys.modules.update({"triton": stub, "triton.language": stub.language})
    try:
        from models.vmamba_vendor import vmamba
    finally:
        del sys.modules["triton"], sys.modules["triton.language"]
    return vmamba


def config(num_classes=3, pretrained=False):
    cfg = ModelConfig()
    cfg.num_classes = num_classes
    cfg.cnn_pretrained = pretrained
    return cfg


@pytest.fixture(scope="module")
def model(vm):
    torch.manual_seed(0)
    return baselines.build_model("vmamba_tiny", config())


def test_vendored_file_is_pinned():
    digest = hashlib.sha256(VENDORED.read_bytes()).hexdigest()
    assert digest == VENDORED_SHA256, "models/vmamba_vendor/vmamba.py differs from the vendored copy"
    text = VENDORED.read_text()
    assert "import selective_scan_cuda" in text  # mamba-ssm's kernel, as upstream
    assert (ROOT / "models" / "vmamba_vendor" / "LICENSE").read_text().startswith("MIT License")


def test_configuration_is_vssm1_tiny_0230s(model):
    b = model.backbone
    assert [len(layer.blocks) for layer in b.layers] == [2, 2, 8, 2]
    assert b.dims == [96, 192, 384, 768]
    assert b.channel_first  # norm_layer ln2d
    for layer, dim in zip(b.layers, b.dims):
        for block in layer.blocks:
            op = block.op
            assert (op.d_state, op.d_inner, op.disable_z) == (1, dim, True)  # ssm_ratio 1, v05_noz
            assert op.forward_core.keywords == {"force_fp32": False, "no_einsum": True}
            assert block.mlp_branch  # mlp_ratio 4
    drop = [blk.drop_path.drop_prob for layer in b.layers for blk in layer.blocks]
    assert drop[0] == 0 and drop[-1] == pytest.approx(0.2)  # the implementation's default


def test_new_head_and_size(model):
    cls = model.backbone.classifier
    assert list(dict(cls.named_children())) == ["norm", "permute", "avgpool", "flatten",
                                                "head_drop", "head"]
    assert cls.head_drop.p == ModelConfig().head_dropout == 0.3
    assert (cls.head.in_features, cls.head.out_features) == (768, 3)
    assert model.count_parameters()["total"] == N_PARAMS_3_CLASSES
    # with the 1000-class layer this is the published ~30 M
    assert N_PARAMS_3_CLASSES + 769 * 997 == 30_249_064


def test_parameter_groups(model):
    groups = model.get_parameter_groups(lr=1e-3, weight_decay=0.05)
    assert [g["name"] for g in groups] == ["backbone", "new_layers"]
    backbone, new = groups
    assert backbone["lr"] == pytest.approx(1e-4) and new["lr"] == pytest.approx(1e-3)
    head = model.backbone.classifier.head
    assert {id(p) for p in new["params"]} == {id(head.weight), id(head.bias)}
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    norm = model.backbone.classifier.norm  # pretrained, so at lr/10
    assert {id(p) for p in norm.parameters()} <= {id(p) for p in backbone["params"]}


@pytest.fixture
def torch_scan(vm, monkeypatch):
    """The authors' PyTorch selective scan, even where the CUDA kernel is installed."""
    monkeypatch.setattr(vm, "WITH_SELECTIVESCAN_MAMBA", False)


def test_forward_backward_cpu(model, torch_scan):
    model.train()
    torch.manual_seed(1)
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 3) and torch.isfinite(logits).all()
    logits.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing[:5]}"
    model.zero_grad(set_to_none=True)


def test_state_dict_round_trip(model, torch_scan):
    # evaluate.py loads checkpoints with strict=False, so a key renamed by
    # VSSM._load_from_state_dict would be skipped silently: check that none is.
    torch.manual_seed(2)
    other = baselines.build_model("vmamba_tiny", config())
    result = other.load_state_dict(model.state_dict(), strict=False)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval(), other.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model(x), other(x))


def test_weights_hash_is_checked(tmp_path, monkeypatch):
    fake = tmp_path / baselines.VMAMBA_T_FILE
    fake.write_bytes(b"not the authors' checkpoint")
    monkeypatch.setattr(baselines, "vmamba_t_weights_path", lambda: fake)
    with pytest.raises(RuntimeError, match="SHA-256"):
        baselines.load_vmamba_t_weights()


def test_pretrained_weights_load_strictly(vm):
    path = Path(torch.hub.get_dir()) / "checkpoints" / baselines.VMAMBA_T_FILE
    if not path.is_file():
        pytest.skip("VMamba-T weights not downloaded here (checked on the GPU instance)")
    m = baselines.build_model("vmamba_tiny", config(pretrained=True))  # strict=True inside
    assert m.count_parameters()["total"] == N_PARAMS_3_CLASSES


def test_registered_for_the_study():
    assert "vmamba_tiny" in baselines.ARCHITECTURES
    study = (ROOT / "scripts" / "run_study.sh").read_text()
    assert '"6|vmamba_tiny|--arch vmamba_tiny"' in study
    assert "*cnn_only*|*vmamba_tiny*) depth_flag=\"\"; gc=\"\"" in study
    queue = (ROOT / "scripts" / "queue_review.sh").read_text()
    assert "VM=vmamba_tiny" in queue and "tools/check_vmamba_kernel.py" in queue
