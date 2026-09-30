"""Registration / difference-image baseline for the EMBED prior-usage audit (label-free, no training).

For every labelled held-out pair the prior view is registered to the current view (affine ECC, initialised from
breast-mask moments), both images are robustly z-scored inside the overlap of the two breast masks, and the
smoothed residual (current - prior) is summarised by tail statistics. The gold change codes are used for evaluation only.

Scores (label-free): the smoothed residual (current - prior) is summarised inside the overlap mask by
  q99   = 99th percentile,  q999 = 99.9th percentile,  mf = maximum of the residual smoothed at lesion scale (sigma = 6 px at 256 px)
for the absolute residual (abs: changed / any change), the positive residual (pos: current brighter, new density) and the
negative residual (neg: prior brighter, resolved density). The statistic that serves as the primary score is chosen from the
synthetic-lesion control only (the gold labels are never used to choose it).
The breast-level score is the maximum over the CC and MLO views. Controls: (i) prior taken from a different patient of the same
split, (ii) a synthetic Gaussian lesion (sigma = 8 px at 256 px, i.e. about 1.5-2 cm across; amplitude 0.2/0.4/0.6 of the [0,1] CLAHE intensity range) added to the current image.

usage (repository root): python audit/registration_baseline.py --out reg_out [--cache cache/prep] [--workers 8] [--limit N]
"""
import argparse, json, os, sys, time
import cv2, numpy as np
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repo root (run scripts from the repo root)
from cbmammo import data as D
from cbmammo import temporal_data as TD
from cbmammo.temporal_concepts import PRESENCE_HEADS, encode_change, encode_new, IGNORE

WORK = 256            # longer image side after downsampling (pixels)
ERODE = 7             # pixels removed from the mask border (skin line, edge artefacts)
SIG_IMG, SIG_RES = 2.0, 3.0

def work_img(a):
    s = WORK / max(a.shape); return cv2.resize(a.astype(np.float32), (max(8, round(a.shape[1] * s)), max(8, round(a.shape[0] * s))), interpolation=cv2.INTER_AREA)

def breast_mask(a):
    m = (gaussian_filter(a, 1.5) > 0.06).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return m.astype(bool)
    return lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))

def moments(m):
    ys, xs = np.nonzero(m); return xs.mean(), ys.mean(), float(m.sum())

def robust_z(x, M):
    v = x[M]; med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) + 1e-6
    return (x - med) / mad

def register_and_score(cur, pri, blob=None, debug=False):
    """cur, pri: float32 images in [0,1] (oriented, CLAHE'd, cropped); returns dict of scores and QC."""
    c, p = work_img(cur), work_img(pri)
    if blob is not None:                       # synthetic lesion added to the current image (control)
        amp, sig = blob; mc0 = breast_mask(c); ys, xs = np.nonzero(mc0)
        if len(xs) == 0: return dict(fail=2)
        k = np.random.default_rng(int(1e6 * c.sum()) % (2**31)).integers(len(xs)); yy, xx = np.mgrid[:c.shape[0], :c.shape[1]]
        c = np.clip(c + amp * np.exp(-((yy - ys[k]) ** 2 + (xx - xs[k]) ** 2) / (2 * sig ** 2)), 0, 1).astype(np.float32)
    mc, mp = breast_mask(c), breast_mask(p)
    if mc.sum() < 500 or mp.sum() < 500: return dict(fail=2)                     # breast mask too small
    (cx, cy, ca), (px, py, pa) = moments(mc), moments(mp); r = np.sqrt(pa / ca)
    W = np.array([[r, 0, px - r * cx], [0, r, py - r * cy]], np.float32)          # current coords -> prior coords
    ok, cc = 0, float("nan")
    try:
        cc, W = cv2.findTransformECC(gaussian_filter(c, 1.0), gaussian_filter(p, 1.0), W, cv2.MOTION_AFFINE,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), mp.astype(np.uint8), 5); ok = 1
    except cv2.error:
        pass
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    pw = cv2.warpAffine(p, W, (c.shape[1], c.shape[0]), flags=flags)
    mw = cv2.warpAffine(mp.astype(np.uint8), W, (c.shape[1], c.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    M = cv2.erode((mc & mw).astype(np.uint8), np.ones((2 * ERODE + 1, 2 * ERODE + 1), np.uint8)) > 0
    if M.sum() < 500: return dict(fail=3)                                          # overlap too small
    res = robust_z(gaussian_filter(c, SIG_IMG), M) - robust_z(gaussian_filter(pw, SIG_IMG), M)
    rs = gaussian_filter(res, SIG_RES); rl = gaussian_filter(res, 6.0)
    v, vl = rs[M], rl[M]
    pc = lambda x, q: float(np.percentile(x, q))
    o = dict(fail=0, ecc=float(cc), ecc_ok=ok, overlap=float(M.sum() / mc.sum()),
             abs_q99=pc(np.abs(v), 99), pos_q99=pc(v, 99), neg_q99=pc(-v, 99),
             abs_q999=pc(np.abs(v), 99.9), pos_q999=pc(v, 99.9), neg_q999=pc(-v, 99.9),
             abs_mf=float(np.abs(vl).max()), pos_mf=float(vl.max()), neg_mf=float((-vl).max()))
    if debug: o["imgs"] = (c, pw, np.where(M, rs, 0.0), M)
    return o

KEYS = ["abs_q99", "pos_q99", "neg_q99", "abs_q999", "pos_q999", "neg_q999", "abs_mf", "pos_mf", "neg_mf", "ecc", "ecc_ok", "overlap", "fail"]
class RegDS(Dataset):
    def __init__(self, base, recs, prior_recs=None, blob=None):
        self.b, self.r, self.prior_recs, self.blob = base, recs, prior_recs, blob
    def __len__(self): return len(self.r)
    def _view(self, views, side, k):
        p = views.get(k)
        return None if p is None else self.b._prep(p, side).astype(np.float32)
    def __getitem__(self, i):
        r = self.r[i]; t = r["embed_temporal"]
        pr = self.prior_recs[i] if self.prior_recs is not None else r                           # record whose PRIOR exam is used
        out = np.full((2, len(KEYS)), np.nan, np.float32)
        for j, k in enumerate(("CC", "MLO")):
            cur = self._view(r["views"], r["side"], k); pri = self._view(pr["embed_temporal"]["prior_views"], pr["side"], k)
            if cur is None or pri is None: out[j, KEYS.index("fail")] = 1; continue                  # missing view
            try:
                o = register_and_score(cur, pri, self.blob)
            except Exception:
                o = dict(fail=4)                                                                    # unexpected error
            out[j] = [o.get(q, np.nan) for q in KEYS]
        return i, out

def run(base, recs, workers, prior_recs=None, blob=None):
    dl = DataLoader(RegDS(base, recs, prior_recs, blob), batch_size=16, shuffle=False, num_workers=workers, collate_fn=lambda b: b)
    res = np.full((len(recs), 2, len(KEYS)), np.nan, np.float32)
    for n, b in enumerate(dl):
        for i, o in b: res[i] = o
        if n % 20 == 0: print(f"  batch {n}/{len(dl)}", flush=True)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/embed_temporal.jsonl"); ap.add_argument("--cache", default="cache/prep")
    ap.add_argument("--out", default="reg_out"); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control_n", type=int, default=200); ap.add_argument("--size", type=int, nargs=2, default=[1036, 644])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    recs = D.load_manifest([a.manifest]); sel = []
    for r in recs:                                                  # same selection as concept_diff_infer.py
        t = r["embed_temporal"]; cl = encode_change(t["change"]); nl, nw = encode_new(t["change"], t["new_candidate"])
        if any(cl[h] != IGNORE for h in PRESENCE_HEADS) or any(nl[h] != IGNORE and nw[h] > 0 for h in PRESENCE_HEADS): sel.append(r)
    sel = [r for r in sel if r["split"] != "train"]                   # training-free baseline: held-out splits only (train rows add nothing)
    if a.limit: sel = sel[: a.limit]
    print(f"selected {len(sel)} labelled held-out pairs", flush=True)
    meta = dict(presence_heads=list(PRESENCE_HEADS), keys=KEYS, sample_id=[r["sample_id"] for r in sel], split=[r["split"] for r in sel],
                patient_id=[str(r["patient_id"]) for r in sel],
                change_labels=[[encode_change(r["embed_temporal"]["change"])[h] for h in PRESENCE_HEADS] for r in sel],
                new_labels=[[encode_new(r["embed_temporal"]["change"], r["embed_temporal"]["new_candidate"])[0][h] for h in PRESENCE_HEADS] for r in sel],
                new_weights=[[encode_new(r["embed_temporal"]["change"], r["embed_temporal"]["new_candidate"])[1][h] for h in PRESENCE_HEADS] for r in sel])
    json.dump(meta, open(f"{a.out}/meta.json", "w"))
    base = TD.TemporalBreastDataset(sel, tuple(a.size), False, True, a.cache, include_weak_new=False)
    true_s = run(base, sel, a.workers); fc = true_s[..., KEYS.index("fail")]
    print(f"true priors done {time.time()-t0:.0f}s; views by failure code (0 ok, 1 missing view, 2 small mask, 3 small overlap, 4 error): " + str({int(k): int((fc == k).sum()) for k in np.unique(fc[~np.isnan(fc)])}) + f"; ECC converged in {np.nanmean(true_s[..., KEYS.index('ecc_ok')]):.3f} of scored views", flush=True)
    panels = []                                                    # diagnostic montage: current | registered prior | smoothed residual, CC view of 6 pairs
    for r in sel[:6]:
        cur = base._prep(r["views"]["CC"], r["side"]).astype(np.float32) if r["views"].get("CC") else None
        pri = base._prep(r["embed_temporal"]["prior_views"]["CC"], r["side"]).astype(np.float32) if r["embed_temporal"]["prior_views"].get("CC") else None
        o = register_and_score(cur, pri, debug=True) if cur is not None and pri is not None else None
        if o and "imgs" in o:
            c_, p_, r_, M_ = o["imgs"]; rr = np.clip(0.5 + r_ / 8.0, 0, 1)
            panels.append(np.concatenate([np.clip(c_, 0, 1), np.clip(p_, 0, 1), rr.astype(np.float32)], 1))
    if panels:
        Hm, Wm = max(p.shape[0] for p in panels), max(p.shape[1] for p in panels); panels = [np.pad(p, ((0, Hm - p.shape[0]), (0, Wm - p.shape[1]))) for p in panels]
        cv2.imwrite(f"{a.out}/diagnostic.png", (np.concatenate(panels, 0) * 255).astype(np.uint8))
    # control 1: prior from a different patient of the same split;  control 2: synthetic lesion added to the current image
    rng = np.random.default_rng(0); ci = rng.choice(len(sel), min(a.control_n, len(sel)), replace=False)
    csel = [sel[i] for i in ci]; prior_recs = []
    for i in ci:
        cand = [r for r in sel if r["split"] == sel[i]["split"] and r["patient_id"] != sel[i]["patient_id"]]
        prior_recs.append(cand[int(rng.integers(len(cand)))])
    mis = run(base, csel, a.workers, prior_recs=prior_recs)
    print(f"mismatched-prior control done {time.time()-t0:.0f}s", flush=True)
    blobs = {}
    for amp in (0.20, 0.40, 0.60):
        blobs[f"blob_{amp:.2f}"] = run(base, csel, a.workers, blob=(amp, 8.0))
    print(f"lesion-insertion control done {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(f"{a.out}/scores.npz", true=true_s, ctrl_index=ci, ctrl_true=true_s[ci], ctrl_mismatched=mis, **blobs)
    print("done", len(sel), "pairs", f"{time.time()-t0:.0f}s", flush=True)
