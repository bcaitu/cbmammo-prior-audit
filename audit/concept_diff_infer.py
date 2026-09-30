"""Paper 2, option B: concept-differencing inference.

Runs FROZEN paper-1 static ConceptModels (cb arm) on the CURRENT and PRIOR exam of every
labelled pair in embed_temporal.jsonl and stores the gated concept-probability vectors for both
timepoints. No training. The change concepts are then formed as differences of these vectors
(analysis is done off-server), and the gold `changed`-code labels are used only for evaluation.

Only records carrying >=1 change label or >=1 non-weak new label are run (others cannot be
scored); train-split rows are kept but flagged by split so they can be reported separately
(paper-1 models saw those breasts during training).
"""
import argparse, json, os, sys, time
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repo root (run scripts from the repo root)
from cbmammo import data as D
from cbmammo import temporal_data as TD
from cbmammo.concepts import HEADS, CONCEPT_HEADS
from cbmammo.model import build, ConceptModel
from cbmammo.temporal_concepts import PRESENCE_HEADS, encode_change, encode_new, IGNORE

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--manifest", default="manifests/embed_temporal.jsonl")
ap.add_argument("--out", default="cd_out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--bs", type=int, default=12)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device("cuda:0")

recs = D.load_manifest([a.manifest])
sel = []
for r in recs:
    t = r["embed_temporal"]
    cl = encode_change(t["change"]); nl, nw = encode_new(t["change"], t["new_candidate"])
    if any(cl[h] != IGNORE for h in PRESENCE_HEADS) or any(nl[h] != IGNORE and nw[h] > 0 for h in PRESENCE_HEADS):
        sel.append(r)
if a.limit:
    sel = sel[: a.limit]
print(f"selected {len(sel)} labelled pairs", flush=True)

def lab(x): return [x.get(h.name, IGNORE) for h in HEADS]
meta = dict(
    heads=[h.name for h in HEADS], presence_heads=list(PRESENCE_HEADS),
    sample_id=[r["sample_id"] for r in sel], split=[r["split"] for r in sel], patient_id=[str(r["patient_id"]) for r in sel],
    side=[r["side"] for r in sel],
    change_labels=[[encode_change(r["embed_temporal"]["change"])[h] for h in PRESENCE_HEADS] for r in sel],
    new_labels=[[encode_new(r["embed_temporal"]["change"], r["embed_temporal"]["new_candidate"])[0][h] for h in PRESENCE_HEADS] for r in sel],
    new_weights=[[encode_new(r["embed_temporal"]["change"], r["embed_temporal"]["new_candidate"])[1][h] for h in PRESENCE_HEADS] for r in sel],
    labels=[lab(r["labels"]) for r in sel], prior_labels=[lab(r["embed_temporal"]["prior_labels"]) for r in sel],
)
json.dump(meta, open(f"{a.out}/meta.json", "w"))
print("meta written", flush=True)

models, size = {}, None
for name in a.runs:
    rd = f"runs/{name}"
    cfg = json.load(open(f"{rd}/run.json"))["config"]
    assert cfg["arm"] == "cb", name
    sz = tuple(cfg["data"]["size"]); size = size or sz
    assert sz == size, (name, sz, size)
    m = build(cfg).to(dev)
    sd = torch.load(f"{rd}/best.pt", map_location=dev); sd = sd.get("state_dict", sd)
    miss, unexp = m.load_state_dict(sd, strict=False)
    print(f"[load] {name}: size={sz} missing={len(miss)} unexpected={len(unexp)}", flush=True)
    assert len(miss) == 0 and len(unexp) == 0, (name, miss[:3], unexp[:3])
    models[name] = m.eval()
print("image size", size, flush=True)

ds = TD.TemporalBreastDataset(sel, size, False, True, "cache/prep", include_weak_new=False)
dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, collate_fn=TD.collate)

cur = {n: [] for n in models}; pri = {n: [] for n in models}; order = []
t0 = time.time()
with torch.no_grad():
    for i, b in enumerate(dl):
        for n, m in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                oc = m(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev))
                op = m(b["prior_cc"].to(dev), b["prior_mlo"].to(dev), b["prior_view_mask"].to(dev))
            cur[n].append(torch.cat([p for p in ConceptModel.concept_probs(oc["logits"]).values()], 1).float().cpu())
            pri[n].append(torch.cat([p for p in ConceptModel.concept_probs(op["logits"]).values()], 1).float().cpu())
        order.append(b["idx"])
        if i % 20 == 0:
            print(f"batch {i}/{len(dl)}  {time.time()-t0:.0f}s", flush=True)

idx = torch.cat(order).numpy()
assert (idx == np.arange(len(sel))).all()
cols = [f"{h.name}:{c}" for h in CONCEPT_HEADS for c in h.classes]
arrs = {}
for n in models:
    arrs[f"cur__{n}"] = torch.cat(cur[n]).numpy(); arrs[f"pri__{n}"] = torch.cat(pri[n]).numpy()
np.savez_compressed(f"{a.out}/probs.npz", **arrs)
meta["columns"] = cols; meta["models"] = list(models)
json.dump(meta, open(f"{a.out}/meta.json", "w"))


print("done", len(sel), "pairs;", len(cols), "concept columns;", f"{time.time()-t0:.0f}s", flush=True)
