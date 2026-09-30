"""Stage 1 model: frozen image encoder -> cross-view fusion -> bottleneck -> heads.

arm = "cb"      : concept bottleneck. Targets (BI-RADS, malignancy) are predicted
                  ONLY from the concept probabilities; the decoder is conditioned
                  only on concepts (see decoder.py).
arm = "opaque"  : matched control. K attention queries -> dense vector z; targets
                  and (auxiliary, weight lambda) concept heads read z. The decoder
                  is conditioned on z.
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concepts import CONCEPT_HEADS, HEAD_BY_NAME, HEADS, IGNORE, TARGET_HEADS


# ------------------------------------------------------------------ encoder
class FrozenEncoder(nn.Module):
    """timm backbone returning a token grid. Works for ViTs (DINOv2/v3, EVA, SigLIP)
    and CNNs (EfficientNet, ConvNeXt, e.g. Mammo-CLIP's EfficientNet-B5 image tower
    via `checkpoint`)."""

    def __init__(self, name: str, pretrained=True, checkpoint: str | None = None,
                 token_pool: int = 2, trainable_blocks: int = 0):
        super().__init__()
        kw = {"dynamic_img_size": True} if name.startswith(("vit", "eva")) else {}
        self.m = timm.create_model(name, pretrained=pretrained and not checkpoint, num_classes=0,
                                   global_pool="", **kw)
        if checkpoint:
            sd = torch.load(checkpoint, map_location="cpu")
            sd = sd.get("state_dict", sd.get("model", sd))
            sd = {k.split("image_encoder.", 1)[-1].replace("model.", "", 1): v for k, v in sd.items()}
            missing, unexpected = self.m.load_state_dict(sd, strict=False)
            print(f"[encoder] loaded {checkpoint}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        self.is_vit = hasattr(self.m, "patch_embed")
        self.dim = self.m.num_features
        self.token_pool = token_pool
        for p in self.m.parameters():
            p.requires_grad = False
        if trainable_blocks and self.is_vit:
            for blk in self.m.blocks[-trainable_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
        self.trainable = trainable_blocks > 0

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable:
            self.m.eval()
        return self

    def forward(self, x):
        with torch.set_grad_enabled(self.trainable and self.training):
            f = self.m.forward_features(x)
        if self.is_vit:
            npre = getattr(self.m, "num_prefix_tokens", 1)
            ph, pw = self.m.patch_embed.patch_size if hasattr(self.m.patch_embed, "patch_size") else (14, 14)
            gh, gw = x.shape[-2] // ph, x.shape[-1] // pw
            f = f[:, npre:, :].transpose(1, 2).reshape(x.shape[0], -1, gh, gw)
        if self.token_pool > 1:
            f = F.avg_pool2d(f, self.token_pool, ceil_mode=True)
        return f                                                   # (B, C, h, w)


def pos2d(h, w, d, device):
    """Fixed 2-D sin/cos positional encoding, (h*w, d)."""
    assert d % 4 == 0
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(d // 4, device=device) / (d // 4)))
    oy, ox = y.flatten()[:, None] * omega, x.flatten()[:, None] * omega
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)


class AttnPool(nn.Module):
    def __init__(self, d, n_queries=1, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_queries, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, tok, pad_mask=None):
        o, w = self.attn(self.q.expand(tok.shape[0], -1, -1), tok, tok, key_padding_mask=pad_mask,
                         need_weights=True, average_attn_weights=True)
        return self.norm(o), w                                    # (B,K,d), (B,K,N)


# ------------------------------------------------------------------ stage-1 model
class ConceptModel(nn.Module):
    def __init__(self, encoder: FrozenEncoder, arm="cb", d=512, fusion_layers=2, heads=8,
                 opaque_queries=8, z_dim=512, cb_detach=True, dropout=0.1):
        super().__init__()
        self.enc, self.arm, self.d, self.cb_detach = encoder, arm, d, cb_detach
        self.proj = nn.Sequential(nn.Linear(encoder.dim, d), nn.LayerNorm(d))
        self.view_emb = nn.Parameter(torch.zeros(2, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, fusion_layers)
        if arm == "cb":
            self.pools = nn.ModuleDict({h.name: AttnPool(d, 1, heads) for h in CONCEPT_HEADS})
            self.clf = nn.ModuleDict({h.name: nn.Linear(d, h.n) for h in CONCEPT_HEADS})
            n_c = sum(h.n for h in CONCEPT_HEADS)
            self.target_mlp = nn.Sequential(nn.Linear(n_c, 128), nn.GELU(), nn.Dropout(dropout))
            self.target_clf = nn.ModuleDict({h.name: nn.Linear(128, h.n) for h in TARGET_HEADS})
        else:
            self.pool = AttnPool(d, opaque_queries, heads)
            self.to_z = nn.Sequential(nn.Linear(opaque_queries * d, z_dim), nn.GELU(), nn.Dropout(dropout))
            self.clf = nn.ModuleDict({h.name: nn.Linear(z_dim, h.n) for h in HEADS})

    # ---- tokens
    def tokens(self, cc, mlo, view_mask):
        B = cc.shape[0]
        f = self.enc(torch.cat([cc, mlo], 0))                     # (2B, C, h, w)
        gap = f.mean((2, 3))                                       # for the linear probe
        h, w = f.shape[-2:]
        t = self.proj(f.flatten(2).transpose(1, 2)) + pos2d(h, w, self.d, f.device)
        t_cc, t_mlo = t[:B] + self.view_emb[0], t[B:] + self.view_emb[1]
        tok = torch.cat([t_cc, t_mlo], 1)
        pad = torch.cat([(~view_mask[:, :1]).expand(-1, h * w), (~view_mask[:, 1:]).expand(-1, h * w)], 1)
        pad[pad.all(1)] = False                                   # never mask everything
        tok = self.fusion(tok, src_key_padding_mask=pad)
        feats = torch.cat([gap[:B], gap[B:]], 1)
        return tok, pad, feats, (h, w)

    # ---- concept probabilities with presence gating (the bottleneck vector)
    @staticmethod
    def concept_probs(logits: dict, override: dict | None = None) -> dict:
        p = {}
        for h in CONCEPT_HEADS:
            p[h.name] = logits[h.name].float().softmax(-1)
            if override and h.name in override:
                p[h.name] = F.one_hot(torch.as_tensor(override[h.name], device=p[h.name].device).expand(
                    p[h.name].shape[0]), h.n).float()
        for h in CONCEPT_HEADS:
            if h.parent:
                p[h.name] = p[h.name] * p[h.parent][:, 1:2]       # absent parent -> ~zero descriptor vector
        return p

    def targets_from_concepts(self, probs: dict) -> dict:
        c = torch.cat([probs[h.name] for h in CONCEPT_HEADS], 1)
        hdn = self.target_mlp(c)
        return {h.name: self.target_clf[h.name](hdn) for h in TARGET_HEADS}

    def forward(self, cc, mlo, view_mask, override=None):
        tok, pad, feats, hw = self.tokens(cc, mlo, view_mask)
        out = {"feats": feats, "grid": hw}
        if self.arm == "cb":
            logits, attn = {}, {}
            for h in CONCEPT_HEADS:
                v, w = self.pools[h.name](tok, pad)
                logits[h.name] = self.clf[h.name](v[:, 0])
                attn[h.name] = w[:, 0]                              # (B, 2*h*w): CC then MLO map
            probs = self.concept_probs(logits, override)
            src = {k: v.detach() for k, v in probs.items()} if self.cb_detach else probs
            logits.update(self.targets_from_concepts(src))
            out.update(logits=logits, attn=attn, cond=torch.cat([probs[h.name] for h in CONCEPT_HEADS], 1))
        else:
            v, w = self.pool(tok, pad)
            z = self.to_z(v.flatten(1))
            out.update(logits={h.name: self.clf[h.name](z) for h in HEADS}, attn={"pool": w.mean(1)}, cond=z)
        return out


def masked_ce(logits: dict, labels: torch.Tensor, weights: dict, class_w: dict | None = None):
    """labels: (B, len(HEADS)) with IGNORE = -1. Returns total loss and per-head losses."""
    total, parts = 0.0, {}
    for j, h in enumerate(HEADS):
        if h.name not in logits or weights.get(h.name, 1.0) == 0:
            continue
        y = labels[:, j]
        m = y != IGNORE
        if m.any():
            cw = class_w.get(h.name) if class_w else None
            l = F.cross_entropy(logits[h.name][m].float(), y[m], weight=cw)
            parts[h.name] = l.detach()
            total = total + weights.get(h.name, 1.0) * l
    return total, parts


def head_weights(arm: str, aux_lambda: float, target_w: float = 1.0) -> dict:
    """cb: concepts and targets weighted 1 (targets see detached concepts by default).
    opaque: targets 1, concepts are an auxiliary task with weight aux_lambda (matched control)."""
    w = {}
    for h in HEADS:
        if h.kind == "target":
            w[h.name] = target_w
        else:
            w[h.name] = 1.0 if arm == "cb" else aux_lambda
    return w


def build(cfg: dict) -> ConceptModel:
    e = cfg["encoder"]
    enc = FrozenEncoder(e["name"], e.get("pretrained", True), e.get("checkpoint"), e.get("token_pool", 2),
                        e.get("trainable_blocks", 0))
    m = cfg["model"]
    return ConceptModel(enc, cfg["arm"], m.get("d", 512), m.get("fusion_layers", 2), m.get("heads", 8),
                        m.get("opaque_queries", 8), m.get("z_dim", 512), m.get("cb_detach", True),
                        m.get("dropout", 0.1))
