"""Image loading, preprocessing and the breast-level (CC + MLO) dataset."""
from __future__ import annotations

import hashlib
import json
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .concepts import HEADS, IGNORE

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_image(path: str) -> np.ndarray:
    """Any of DICOM / PNG / JPEG -> float32 grayscale in [0, 1], MONOCHROME1 inverted."""
    if path.lower().endswith((".dcm", ".dicom")) or os.path.splitext(path)[1] == "":
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut
        d = pydicom.dcmread(path)
        a = d.pixel_array.astype(np.float32)
        try:
            a = apply_voi_lut(a, d).astype(np.float32)
        except Exception:
            pass
        if str(getattr(d, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = a.max() - a
    else:
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(path)
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        a = a.astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def orient_and_crop(a: np.ndarray, side: str | None) -> np.ndarray:
    """Flip so the chest wall is on the left, then crop the dark background around the breast."""
    h, w = a.shape
    if a[:, : w // 2].mean() < a[:, w // 2:].mean():
        a = a[:, ::-1]
    mask = (a > 0.05).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, ww, hh = stats[k, :4]
        a = a[y:y + hh, x:x + ww]
    return np.ascontiguousarray(a)


def clahe(a: np.ndarray, clip=2.0) -> np.ndarray:
    u8 = (a * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255


def to_tensor(a: np.ndarray, size: tuple, train: bool) -> torch.Tensor:
    """Aspect-preserving resize + pad to (H, W); light affine jitter in training."""
    H, W = size
    h, w = a.shape
    s = min(H / h, W / w)
    a = cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    if train:
        ang, sc = random.uniform(-7, 7), random.uniform(0.95, 1.05)
        M = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), ang, sc)
        a = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), borderValue=0)
        if random.random() < 0.5:
            a = a[::-1]                                           # vertical flip keeps chest-wall side
    out = np.zeros((H, W), np.float32)
    out[: a.shape[0], : a.shape[1]] = a
    x = np.repeat(out[..., None], 3, -1)
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def load_manifest(paths, splits=None, datasets=None) -> list:
    if isinstance(paths, str):
        paths = [paths]
    recs = []
    for p in paths:
        with open(p) as f:
            recs += [json.loads(l) for l in f if l.strip()]
    if splits:
        recs = [r for r in recs if r["split"] in splits]
    if datasets:
        recs = [r for r in recs if r["dataset"] in datasets]
    return [r for r in recs if any(r["views"].values())]


class BreastDataset(Dataset):
    """Returns both views (a missing view is replaced by the other and flagged)."""

    def __init__(self, records, size=(1024, 640), train=False, use_clahe=True, cache_dir=None):
        self.r, self.size, self.train, self.use_clahe, self.cache = records, tuple(size), train, use_clahe, cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.r)

    def _prep(self, path, side):
        if self.cache:
            key = hashlib.md5(f"{path}|{self.size}|{self.use_clahe}".encode()).hexdigest()
            cp = os.path.join(self.cache, f"{key}.npy")
            if os.path.exists(cp):
                try:
                    return np.load(cp)
                except Exception:          # truncated by a concurrent writer -> recompute
                    pass
        a = orient_and_crop(read_image(path), side)
        if self.use_clahe:
            a = clahe(a)
        # cache at 2x the training size to keep disk use bounded
        H, W = self.size
        s = min(2 * H / a.shape[0], 2 * W / a.shape[1], 1.0)
        a = cv2.resize(a, (int(a.shape[1] * s), int(a.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if self.cache:
            tmp = f"{cp}.{os.getpid()}.tmp.npy"
            np.save(tmp, a.astype(np.float16))
            os.replace(tmp, cp)                   # atomic: readers never see a partial file
        return a

    def __getitem__(self, i):
        r = self.r[i]
        v = r["views"]
        cc_p, mlo_p = v.get("CC") or v.get("MLO"), v.get("MLO") or v.get("CC")
        cc = to_tensor(self._prep(cc_p, r["side"]).astype(np.float32), self.size, self.train)
        mlo = to_tensor(self._prep(mlo_p, r["side"]).astype(np.float32), self.size, self.train)
        labels = torch.tensor([r["labels"][h.name] for h in HEADS], dtype=torch.long)
        return {"cc": cc, "mlo": mlo, "labels": labels,
                "view_mask": torch.tensor([v.get("CC") is not None, v.get("MLO") is not None]),
                "idx": i}


def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in ("cc", "mlo", "labels", "view_mask")}
    out["idx"] = torch.tensor([x["idx"] for x in b])
    return out


def epoch_sample(records, embed_neg_ratio: float | None, seed: int = 0) -> list:
    """Per-epoch training sample: every non-EMBED record and every EMBED record with a finding,
    BI-RADS >= 2 or known pathology, plus a freshly-drawn sample of EMBED BI-RADS-1/no-finding
    breasts at `embed_neg_ratio` negatives per such EMBED record (round-half-up, capped at the
    number available). Call once per epoch with a different `seed` so the negative pool rotates;
    over enough epochs almost all EMBED negatives are seen, just not every one every epoch.
    `embed_neg_ratio=None` returns `records` unchanged (pre-EMBED behaviour, exact reproduction).
    """
    if embed_neg_ratio is None or not any(r["dataset"] == "embed" for r in records):
        return records

    def is_embed_negative(r):
        if r["dataset"] != "embed":
            return False
        v = r["values"]
        return v.get("birads") == "1" and all(v.get(p) in (None, "absent") for p in ("mass", "calc", "asymmetry", "distortion"))

    other = [r for r in records if not is_embed_negative(r)]
    neg = [r for r in records if is_embed_negative(r)]
    n_embed_other = sum(1 for r in other if r["dataset"] == "embed")
    n_take = min(len(neg), round(n_embed_other * embed_neg_ratio))
    rng = random.Random(seed)
    picked = rng.sample(neg, n_take) if n_take < len(neg) else list(neg)
    return other + picked


def label_counts(records) -> dict:
    """Per head, per class counts (for class weights and the feasibility table)."""
    out = {}
    for h in HEADS:
        c = np.zeros(h.n, int)
        for r in records:
            if r["labels"][h.name] != IGNORE:
                c[r["labels"][h.name]] += 1
        out[h.name] = c
    return out
