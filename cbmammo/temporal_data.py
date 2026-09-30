"""Paper 2: paired current+prior dataset, reading embed_temporal.jsonl-shaped records.

Subclasses data.BreastDataset so the current-exam path (image prep, augmentation, caching,
current-state labels) is reused UNCHANGED -- this file only adds the prior-exam image pair and
the change/new-detection label tensors on top. load_manifest() from data.py needs no changes
(embed_temporal.jsonl records already carry 'views'/'labels'/'split'/'dataset' like every other
manifest; the extra 'embed_temporal' dict is just additional keys load_manifest already passes
through unchanged).
"""
from __future__ import annotations

import numpy as np
import torch

from .data import BreastDataset, to_tensor, collate as base_collate
from .temporal_concepts import TEMPORAL_HEADS, encode_change, encode_new


class TemporalBreastDataset(BreastDataset):
    def __init__(self, *args, include_weak_new: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_weak_new = include_weak_new

    def _views(self, views: dict, side: str):
        """Same missing-view fallback as BreastDataset.__getitem__ (fall back to the other view
        of the SAME exam), factored out so it can be reused for both the current and prior pair."""
        cc_p, mlo_p = views.get("CC") or views.get("MLO"), views.get("MLO") or views.get("CC")
        cc = to_tensor(self._prep(cc_p, side).astype(np.float32), self.size, self.train)
        mlo = to_tensor(self._prep(mlo_p, side).astype(np.float32), self.size, self.train)
        mask = torch.tensor([views.get("CC") is not None, views.get("MLO") is not None])
        return cc, mlo, mask

    def __getitem__(self, i):
        r = self.r[i]
        t = r["embed_temporal"]
        out = super().__getitem__(i)                        # cc, mlo, labels, view_mask, idx (current exam)
        p_cc, p_mlo, pvm = self._views(t["prior_views"], r["side"])

        change_lab = encode_change(t["change"])
        new_lab, new_w = encode_new(t["change"], t["new_candidate"], include_weak=self.include_weak_new)
        out.update(
            prior_cc=p_cc, prior_mlo=p_mlo, prior_view_mask=pvm,
            change_labels=torch.tensor([change_lab[h] for h in TEMPORAL_HEADS], dtype=torch.long),
            new_labels=torch.tensor([new_lab[h] for h in TEMPORAL_HEADS], dtype=torch.long),
            new_weights=torch.tensor([new_w[h] for h in TEMPORAL_HEADS], dtype=torch.float32),
        )
        return out


def collate(b):
    out = base_collate(b)
    extra = ("prior_cc", "prior_mlo", "prior_view_mask", "change_labels", "new_labels", "new_weights")
    out.update({k: torch.stack([x[k] for x in b]) for k in extra})
    return out
