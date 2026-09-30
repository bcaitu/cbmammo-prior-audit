"""Prior-usage controls of the audit, tested on a tiny randomly initialised encoder (no downloads, CPU only)."""
import pytest
import torch

from cbmammo.model import ConceptModel, FrozenEncoder
from cbmammo.temporal_concepts import TEMPORAL_HEADS, IGNORE
from cbmammo.temporal_model import OpaqueTemporalModel, TemporalConceptModel, temporal_loss


def _base(arm):
    torch.manual_seed(0)
    enc = FrozenEncoder("vit_tiny_patch16_224", pretrained=False, token_pool=2)
    return ConceptModel(enc, arm=arm, d=32, fusion_layers=1, heads=4, opaque_queries=2, z_dim=16)


def _images(seed, b=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 3, 64, 48, generator=g), torch.randn(b, 3, 64, 48, generator=g), torch.ones(b, 2, dtype=torch.bool)


def _wrap(arm, mode):
    cls = TemporalConceptModel if arm == "cb" else OpaqueTemporalModel
    return cls(_base(arm), heads=4, dropout=0.0, prior_mode=mode).eval()


def _heads(out):
    return torch.cat([out["change_logits"][h] for h in TEMPORAL_HEADS] + [out["new_logits"][h] for h in TEMPORAL_HEADS], 1)


@pytest.mark.parametrize("arm", ["cb", "opaque"])
def test_current_only_control_ignores_the_prior(arm):
    m = _wrap(arm, "self")
    cc, mlo, vm = _images(1)
    with torch.no_grad():
        a = _heads(m(cc, mlo, vm, *_images(2)))
        b = _heads(m(cc, mlo, vm, *_images(3)))
    assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("arm", ["cb", "opaque"])
def test_prior_model_depends_on_the_prior(arm):
    m = _wrap(arm, "prior")
    cc, mlo, vm = _images(1)
    with torch.no_grad():
        a = _heads(m(cc, mlo, vm, *_images(2)))
        b = _heads(m(cc, mlo, vm, *_images(3)))
    assert not torch.allclose(a, b, atol=1e-4)


@pytest.mark.parametrize("arm", ["cb", "opaque"])
def test_current_only_control_has_identical_parameter_count(arm):
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(_wrap(arm, "self")) == n(_wrap(arm, "prior"))


def test_loss_without_any_label_is_differentiable():
    m = _wrap("cb", "prior").train()
    cc, mlo, vm = _images(1)
    out = m(cc, mlo, vm, *_images(2))
    k = len(TEMPORAL_HEADS)
    ign = torch.full((2, k), IGNORE, dtype=torch.long)
    loss, parts = temporal_loss(out, ign, ign, torch.zeros(2, k))
    assert parts == {} and loss.requires_grad
    loss.backward()
