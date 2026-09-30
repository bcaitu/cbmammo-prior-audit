"""Paper 2 control: do the trained temporal heads actually USE the prior exam?
For every labelled held-out pair, score the trained heads with (a) the TRUE prior, (b) a prior taken from a
DIFFERENT patient in the same split (shuffled), (c) the current exam fed as its own prior (identity: zero interval change).
If AUROC(true) ~= AUROC(shuffled) the head is a current-exam-only classifier.
usage: python prior_shuffle_eval.py <out_dir> run1 run2 ..."""
import json, os, sys, time, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repo root (run scripts from the repo root)
from cbmammo import data as D, temporal_data as TD
from cbmammo.train_stage1_temporal import build_temporal, is_labeled
from cbmammo.temporal_concepts import TEMPORAL_HEADS
out_dir, runs = sys.argv[1], sys.argv[2:]; os.makedirs(out_dir, exist_ok=True)
dev = torch.device("cuda:0")
recs = D.load_manifest(["manifests/embed_temporal.jsonl"])
SPL = ["val", "test", "test_embed_ge", "test_embed_fuji"]
sel = [r for r in recs if r["split"] in SPL and is_labeled(r)]
print("held-out labelled pairs:", len(sel), flush=True)
rng = np.random.default_rng(0); perm = np.arange(len(sel))
for sp in SPL:
    ii = np.array([i for i, r in enumerate(sel) if r["split"] == sp]); pids = np.array([str(sel[i]["patient_id"]) for i in ii])
    p = rng.permutation(len(ii))
    for _ in range(200):
        bad = np.where(pids[p] == pids)[0]
        if len(bad) == 0: break
        for b in bad:
            k = rng.integers(len(ii)); p[b], p[k] = p[k], p[b]
    assert (pids[p] != pids).all(), "could not derange patients in " + sp
    perm[ii] = ii[p]
models, size = {}, None
for name in runs:
    cfg = json.load(open(f"runs/{name}/run.json"))["config"]
    m = build_temporal(cfg, dev)
    sd = torch.load(f"runs/{name}/best.pt", map_location=dev)
    miss, unexp = m.load_state_dict(sd, strict=False); print(name, "missing", len(miss), "unexpected", len(unexp), flush=True)
    assert len(miss) == 0 and len(unexp) == 0
    models[name] = m.eval(); size = tuple(cfg["data"]["size"])
ds = TD.TemporalBreastDataset(sel, size, False, True, "cache/prep", include_weak_new=False)
class Sh(Dataset):
    def __len__(self): return len(ds)
    def __getitem__(self, i):
        a = ds[i]; b = ds[int(perm[i])]
        a["sh_cc"], a["sh_mlo"], a["sh_vm"] = b["prior_cc"], b["prior_mlo"], b["prior_view_mask"]; return a
def coll(bt):
    o = TD.collate(bt)
    for k in ("sh_cc", "sh_mlo", "sh_vm"): o[k] = torch.stack([x[k] for x in bt])
    return o
dl = DataLoader(Sh(), batch_size=12, shuffle=False, num_workers=12, collate_fn=coll)
acc = {n: {c: {"chg": [], "new": []} for c in ("true", "shuf", "ident")} for n in models}
t0 = time.time()
with torch.no_grad():
    for bi, b in enumerate(dl):
        cc, mlo, vm = b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev)
        conds = {"true": (b["prior_cc"], b["prior_mlo"], b["prior_view_mask"]), "shuf": (b["sh_cc"], b["sh_mlo"], b["sh_vm"]), "ident": (b["cc"], b["mlo"], b["view_mask"])}
        for n, m in models.items():
            for c, (pc, pm, pv) in conds.items():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o = m(cc, mlo, vm, pc.to(dev), pm.to(dev), pv.to(dev))
                acc[n][c]["chg"].append(torch.stack([o["change_logits"][h].float().softmax(-1) for h in TEMPORAL_HEADS], 1).cpu())
                acc[n][c]["new"].append(torch.stack([o["new_logits"][h].float().softmax(-1) for h in TEMPORAL_HEADS], 1).cpu())
        if bi % 20 == 0: print(f"batch {bi}/{len(dl)} {time.time()-t0:.0f}s", flush=True)
arrs = {}
for n in models:
    for c in ("true", "shuf", "ident"):
        arrs[f"{n}__{c}__chg"] = torch.cat(acc[n][c]["chg"]).numpy(); arrs[f"{n}__{c}__new"] = torch.cat(acc[n][c]["new"]).numpy()
np.savez_compressed(f"{out_dir}/prior_shuffle_probs.npz", **arrs)
json.dump(dict(sample_id=[r["sample_id"] for r in sel], donor=[sel[int(j)]["sample_id"] for j in perm], runs=list(models), heads=list(TEMPORAL_HEADS)), open(f"{out_dir}/prior_shuffle_meta.json", "w"))
print("done", time.time() - t0, flush=True)
