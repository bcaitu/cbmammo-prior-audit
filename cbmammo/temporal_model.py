"""Paper 2: temporal (current + prior exam) extension of the stage-1 ConceptModel.

Reuses paper 1's ConceptModel wholesale (encoder, per-view fusion, concept heads, diagnosis
head) for BOTH the current and prior exam's own token sequence, then adds one new
cross-attention block letting current tokens attend to prior tokens. Nothing about paper 1's
own forward pass changes -- this module only ADDS a temporal path on top, matching the
brief's reuse-heavy design (paper2_brief_temporal_change.md section 5).

The module is exercised by tests/test_temporal_prior.py (prior-independence of the current-only control) and was
used for all reported runs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ConceptModel, AttnPool
from .concepts import CONCEPT_HEADS
from .temporal_concepts import TEMPORAL_HEADS, CHANGE_CLASSES, NEW_CLASSES, IGNORE as T_IGNORE


class TemporalCrossAttn(nn.Module):
    """Current tokens attend to prior tokens (cross-attention, current tokens = query)."""

    def __init__(self, d, heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=dropout)
        self.norm_q, self.norm_kv, self.norm_out = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, tok_cur, pad_cur, tok_prior, pad_prior):
        q, kv = self.norm_q(tok_cur), self.norm_kv(tok_prior)
        # if a sample's prior tokens are ALL padding (both prior views missing -- shouldn't
        # happen given the manifest builder requires a resolvable prior image, but guard anyway
        # since MultiheadAttention raises on an all-True padding row), fall back to no masking.
        safe_pad_prior = pad_prior.clone()
        safe_pad_prior[safe_pad_prior.all(1)] = False
        out, _ = self.attn(q, kv, kv, key_padding_mask=safe_pad_prior)
        return self.norm_out(tok_cur + out)                        # residual


class TemporalConceptModel(nn.Module):
    """Wraps a base ConceptModel(arm='cb'), reused UNMODIFIED for the current-state concept
    heads and diagnosis head (from tok_cur, not the temporally-fused tokens -- this keeps
    paper 1's leakage-free current-state path identical, independent of the prior exam);
    adds temporal cross-attention + change/new-detection heads on top."""

    def __init__(self, base: ConceptModel, heads=8, dropout=0.1, prior_mode="prior"):
        super().__init__()
        assert base.arm == "cb", "temporal change heads are defined on top of the concept-bottleneck arm"
        assert prior_mode in ("prior", "self")
        # prior_mode='self' is the CURRENT-EXAM-ONLY control: the same attention block (same parameter
        # count and compute) attends to the current exam's own tokens, so no prior information can enter.
        self.prior_mode = prior_mode
        self.base = base
        d = base.d
        self.cross = TemporalCrossAttn(d, heads, dropout)
        self.change_pools = nn.ModuleDict({h: AttnPool(d, 1, heads) for h in TEMPORAL_HEADS})
        self.change_clf = nn.ModuleDict({h: nn.Linear(d, len(CHANGE_CLASSES)) for h in TEMPORAL_HEADS})
        self.new_clf = nn.ModuleDict({h: nn.Linear(d, len(NEW_CLASSES)) for h in TEMPORAL_HEADS})

    def train(self, mode=True):
        """Mirrors FrozenEncoder.train(): if every base param is frozen, keep base in eval mode
        (dropout/etc off) even when the wrapper (this module's new submodules) is training."""
        super().train(mode)
        if not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        return self

    def forward(self, cc, mlo, view_mask, prior_cc, prior_mlo, prior_view_mask):
        tok_cur, pad_cur, _, _ = self.base.tokens(cc, mlo, view_mask)
        if self.prior_mode == "self":
            tok_prior, pad_prior = tok_cur, pad_cur                  # current-exam-only control
        else:
            tok_prior, pad_prior, _, _ = self.base.tokens(prior_cc, prior_mlo, prior_view_mask)
        tok_fused = self.cross(tok_cur, pad_cur, tok_prior, pad_prior)

        logits, attn = {}, {}
        for h in CONCEPT_HEADS:
            v, w = self.base.pools[h.name](tok_cur, pad_cur)
            logits[h.name] = self.base.clf[h.name](v[:, 0])
            attn[h.name] = w[:, 0]
        probs = ConceptModel.concept_probs(logits)
        src = {k: v.detach() for k, v in probs.items()} if self.base.cb_detach else probs
        logits.update(self.base.targets_from_concepts(src))

        change_logits, new_logits = {}, {}
        for h in TEMPORAL_HEADS:
            v, _ = self.change_pools[h](tok_fused, pad_cur)
            change_logits[h] = self.change_clf[h](v[:, 0])
            new_logits[h] = self.new_clf[h](v[:, 0])
        return {"logits": logits, "attn": attn, "change_logits": change_logits, "new_logits": new_logits,
                "cond": torch.cat([probs[h.name] for h in CONCEPT_HEADS], 1)}


class OpaqueTemporalModel(nn.Module):
    """Matched opaque-temporal control: same cross-attention block, but the change/new heads
    AND the diagnosis targets read a single pooled temporal latent z instead of change-concepts
    -- mirrors how ConceptModel's own opaque arm reads one pooled z instead of per-head concepts.
    `base` must be a ConceptModel(arm='opaque'). Change/new heads here are auxiliary (weighted by
    aux_lambda in the loss, same convention as model.head_weights), never feeding the diagnosis."""

    def __init__(self, base: ConceptModel, heads=8, dropout=0.1, prior_mode="prior"):
        super().__init__()
        assert base.arm == "opaque", "OpaqueTemporalModel wraps a ConceptModel(arm='opaque')"
        assert prior_mode in ("prior", "self")
        self.prior_mode = prior_mode
        self.base = base
        z_dim = base.to_z[0].out_features            # ConceptModel doesn't keep z_dim as an attribute
        self.cross = TemporalCrossAttn(base.d, heads, dropout)
        self.change_clf = nn.ModuleDict({h: nn.Linear(z_dim, len(CHANGE_CLASSES)) for h in TEMPORAL_HEADS})
        self.new_clf = nn.ModuleDict({h: nn.Linear(z_dim, len(NEW_CLASSES)) for h in TEMPORAL_HEADS})

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        return self

    def forward(self, cc, mlo, view_mask, prior_cc, prior_mlo, prior_view_mask):
        tok_cur, pad_cur, _, _ = self.base.tokens(cc, mlo, view_mask)
        if self.prior_mode == "self":
            tok_prior, pad_prior = tok_cur, pad_cur                  # current-exam-only control
        else:
            tok_prior, pad_prior, _, _ = self.base.tokens(prior_cc, prior_mlo, prior_view_mask)
        tok_fused = self.cross(tok_cur, pad_cur, tok_prior, pad_prior)
        v, _ = self.base.pool(tok_fused, pad_cur)
        z = self.base.to_z(v.flatten(1))
        logits = {name: self.base.clf[name](z) for name in self.base.clf}
        change_logits = {h: self.change_clf[h](z) for h in TEMPORAL_HEADS}
        new_logits = {h: self.new_clf[h](z) for h in TEMPORAL_HEADS}
        return {"logits": logits, "change_logits": change_logits, "new_logits": new_logits, "cond": z}


def temporal_loss(out: dict, change_labels: torch.Tensor, new_labels: torch.Tensor,
                   new_weights: torch.Tensor, aux_lambda: float = 1.0,
                   change_cw: dict | None = None, new_cw: dict | None = None) -> tuple:
    """change_labels/new_labels: (B, len(TEMPORAL_HEADS)) long, T_IGNORE where unlabelled.
    new_weights: (B, len(TEMPORAL_HEADS)) float, per-example loss weight (0 excludes -- this is
    how a 'weak' new_candidate tag is dropped from the loss by default, see temporal_concepts.py).
    aux_lambda scales both terms down for the opaque-temporal control (matched-cost convention);
    leave at 1.0 for the concept-bottleneck arm.
    change_cw/new_cw: optional {head: tensor} per-class weight (from temporal_class_weights),
    passed to F.cross_entropy the same way model.masked_ce does for paper 1's heads -- WITHOUT
    this, cross-entropy on the raw (severely imbalanced, e.g. 10-230 labelled val examples out
    of thousands) label distribution predictably collapses to majority-class prediction (this
    was found and fixed 2026-09-27 after the first full run's val predictions turned out
    constant -- e.g. new_mass predicted class 1 for ALL 5592 val examples, not just the ~150
    labelled ones).

    A batch can legitimately have ZERO labelled change/new examples across every head (small
    batch, sparse labels) -- `total` starts at None rather than 0.0 so that case still returns
    a tensor connected to the graph (any real model output times 0), not a plain float, which
    would crash `.backward()` with 'float object has no attribute backward'."""
    total, parts = None, {}
    for j, h in enumerate(TEMPORAL_HEADS):
        yc = change_labels[:, j]
        m = yc != T_IGNORE
        if m.any():
            cw = change_cw.get(h) if change_cw else None
            l = F.cross_entropy(out["change_logits"][h][m].float(), yc[m], weight=cw)
            parts[f"change_{h}"] = l.detach()
            term = aux_lambda * l
            total = term if total is None else total + term
        yn, wn = new_labels[:, j], new_weights[:, j]
        m2 = (yn != T_IGNORE) & (wn > 0)
        if m2.any():
            l2 = F.cross_entropy(out["new_logits"][h][m2].float(), yn[m2], reduction="none")
            cwn = new_cw.get(h) if new_cw else None
            combined_w = wn[m2] * (cwn[yn[m2]] if cwn is not None else 1.0)
            l2 = (l2 * combined_w).sum() / combined_w.sum()
            parts[f"new_{h}"] = l2.detach()
            term2 = aux_lambda * l2
            total = term2 if total is None else total + term2
    if total is None:
        total = sum(v.sum() for v in out["change_logits"].values()) * 0.0
    return total, parts
