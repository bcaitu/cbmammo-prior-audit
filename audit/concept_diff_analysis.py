"""Paper 2 option B analysis: label-free concept-differencing vs gold change labels.
Change concepts = differences of frozen paper-1 static concept probabilities at t and t-1
(ensemble mean of 3 seeds). Gold `changed`-code labels are used for EVALUATION ONLY."""
import json, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
D = sys.argv[1]; OUT = sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
meta = json.load(open(f"{D}/meta.json")); P = np.load(f"{D}/probs.npz")
cols, models = meta["columns"], meta["models"]
cur = np.mean([P[f"cur__{m}"] for m in models], 0); pri = np.mean([P[f"pri__{m}"] for m in models], 0)
per_seed = {m: (P[f"cur__{m}"], P[f"pri__{m}"]) for m in models}
PH = meta["presence_heads"]; HN = meta["heads"]
cl = np.array(meta["change_labels"]); nl = np.array(meta["new_labels"]); nw = np.array(meta["new_weights"])
gold_now = np.array(meta["labels"]); gold_pri = np.array(meta["prior_labels"])
split = np.array(meta["split"]); pid = np.array(meta["patient_id"])
groups = {"mass": ["mass", "mass_shape", "mass_margin"], "calc": ["calc", "calc_morphology", "calc_distribution"],
          "asymmetry": ["asymmetry"], "distortion": ["distortion"]}
cidx = {c: i for i, c in enumerate(cols)}
def gi(h): return [i for c, i in cidx.items() if c.split(":")[0] in groups[h]]
def sc(cur, pri):
    o = {}
    for h in PH:
        pp = cidx[f"{h}:present"]
        o[h] = dict(chg=0.5 * np.abs(cur[:, gi(h)] - pri[:, gi(h)]).sum(1), drop=pri[:, pp] - cur[:, pp],
                    rise=cur[:, pp] - pri[:, pp], level=cur[:, pp])
    o["pooled"] = {k: np.max([o[h][k] for h in PH], 0) for k in ("chg", "drop", "rise", "level")}
    return o
S = sc(cur, pri); SS = {m: sc(*per_seed[m]) for m in models}
def gold_sc():
    o = {}
    for h in PH:
        j = HN.index(h); a, b = gold_now[:, j].astype(float), gold_pri[:, j].astype(float)
        a[a < 0] = np.nan; b[b < 0] = np.nan
        o[h] = dict(chg=np.abs(a - b), drop=b - a, rise=a - b, level=a)
    o["pooled"] = {k: np.nanmax(np.stack([o[h][k] for h in PH]), 0) if True else None for k in ("chg", "drop", "rise", "level")}
    return o
import warnings; warnings.filterwarnings("ignore")
G = gold_sc()
# labels per unit
def pooled_change(r):
    v = set(x for x in r if x >= 0); return 2 if 2 in v else 1 if 1 in v else 0 if 0 in v else -1
def pooled_new(rl, rw):
    if any(l == 1 and w > 0 for l, w in zip(rl, rw)): return 1
    return 0 if any(l == 0 for l in rl) else -1
Y = {h: cl[:, j] for j, h in enumerate(PH)}; Y["pooled"] = np.array([pooled_change(r) for r in cl])
N = {h: np.where(nw[:, j] > 0, nl[:, j], -1) for j, h in enumerate(PH)}; N["pooled"] = np.array([pooled_new(a, b) for a, b in zip(nl, nw)])
def auroc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
TASKS = {  # name: (label source, keep, positive, score key, control sign)
    "changed_vs_stable": ("c", lambda y: np.isin(y, [0, 2]), lambda y: y == 2, "chg", +1),
    "resolved_vs_stable": ("c", lambda y: np.isin(y, [0, 1]), lambda y: y == 1, "drop", -1),
    "anychange_vs_stable": ("c", lambda y: y >= 0, lambda y: y > 0, "chg", +1),
    "new_vs_notnew(silver)": ("n", lambda y: y >= 0, lambda y: y == 1, "rise", +1),
}
GROUPS = {"heldout": ["val", "test", "test_embed_ge", "test_embed_fuji"], "val": ["val"], "test": ["test"],
          "test_embed_ge": ["test_embed_ge"], "test_embed_fuji": ["test_embed_fuji"], "train(seen by static model)": ["train"]}
rng = np.random.default_rng(0); rows = []
for gname, sp in GROUPS.items():
    gm = np.isin(split, sp)
    for u in ["mass", "calc", "asymmetry", "distortion", "pooled"]:
        for tname, (src, keep, pos, key, csign) in TASKS.items():
            y = (Y if src == "c" else N)[u]; m = gm & keep(y)
            n1, n0 = int(pos(y[m]).sum()), int((~pos(y[m])).sum())
            row = dict(group=gname, unit=u, task=tname, n_pos=n1, n_neg=n0)
            if n1 >= 1 and n0 >= 1:
                idx = np.where(m)[0]; yy = pos(y[idx]); s = S[u][key][idx]
                row["auroc"] = auroc(yy, s)
                row["auroc_level_only"] = auroc(yy, csign * S[u]["level"][idx] if csign != -1 else -S[u]["level"][idx])
                sa = [auroc(yy, SS[mm][u][key][idx]) for mm in models]; row["seed_min"], row["seed_max"] = min(sa), max(sa)
                gs = G[u][key][idx]; ok = ~np.isnan(gs)
                row["n_oracle"] = int(ok.sum()); row["auroc_oracle_presence"] = auroc(yy[ok], gs[ok]) if ok.sum() > 3 else np.nan
                if n1 >= 5 and n0 >= 5:
                    up = {p: np.where(pid[idx] == p)[0] for p in np.unique(pid[idx])}; keys = list(up)
                    bs = []
                    for _ in range(B):
                        ii = np.concatenate([up[k] for k in rng.choice(keys, len(keys))])
                        a = auroc(yy[ii], s[ii]); bs.append(a)
                    bs = np.array(bs); bs = bs[~np.isnan(bs)]
                    row["ci_lo"], row["ci_hi"] = np.percentile(bs, [2.5, 97.5])
                    row["n_patients"] = len(keys)
            rows.append(row)
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
show = df[(df.group == "heldout")][["unit", "task", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi", "seed_min", "seed_max", "auroc_level_only", "auroc_oracle_presence", "n_oracle"]]
print(show.round(3).to_string(index=False))
