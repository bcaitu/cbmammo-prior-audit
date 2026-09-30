
import json, glob, sys, numpy as np, pandas as pd, warnings
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
D = sys.argv[1]; CD = sys.argv[2]; OUT = sys.argv[3]; B = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
ps = json.load(open(f"{D}/prior_shuffle_meta.json")); Z = np.load(f"{D}/prior_shuffle_probs.npz"); cdm = json.load(open(f"{CD}/meta.json"))
pos = {s: i for i, s in enumerate(cdm["sample_id"])}; ix = np.array([pos[s] for s in ps["sample_id"]])
PH = ["mass", "calc", "asymmetry", "distortion"]; UN = PH + ["pooled"]
cl = np.array(cdm["change_labels"])[ix]; nl = np.array(cdm["new_labels"])[ix]; nw = np.array(cdm["new_weights"])[ix]
split = np.array(cdm["split"])[ix]; pid = np.array(cdm["patient_id"])[ix]
def pc(r): v = set(x for x in r if x >= 0); return 2 if 2 in v else 1 if 1 in v else 0 if 0 in v else -1
def pn(a, b): return 1 if any(l == 1 and w > 0 for l, w in zip(a, b)) else (0 if any(l == 0 for l in a) else -1)
Y = {h: cl[:, j] for j, h in enumerate(PH)}; Y["pooled"] = np.array([pc(r) for r in cl])
N = {h: np.where(nw[:, j] > 0, nl[:, j], -1) for j, h in enumerate(PH)}; N["pooled"] = np.array([pn(a, b) for a, b in zip(nl, nw)])
runs = ps["runs"]; arms = {"CB": [r for r in runs if r.startswith("cb_")], "OPQ": [r for r in runs if r.startswith("opq_")]}
def ens(arm, cond, kind): return np.mean([Z[f"{r}__{cond}__{kind}"] for r in arms[arm]], 0)   # (N, 5, C)
TASKS = {"changed_vs_stable": ("c", [0, 2], 2, lambda P: P[..., 2]), "resolved_vs_stable": ("c", [0, 1], 1, lambda P: P[..., 1]),
         "anychange_vs_stable": ("c", [0, 1, 2], None, lambda P: 1 - P[..., 0]), "new_vs_notnew(silver)": ("n", [0, 1], 1, lambda P: P[..., 1])}
def au(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
rng = np.random.default_rng(1); rows = []
for arm in arms:
    if not arms[arm]: continue
    for sp in ["test", "test_embed_ge", "test_embed_fuji", "val"]:
        for u in UN:
            j = UN.index(u)
            for t, (src, keep, posc, f) in TASKS.items():
                y = (Y if src == "c" else N)[u]; m = (split == sp) & np.isin(y, keep); idx = np.where(m)[0]
                p = (y[idx] > 0) if posc is None else (y[idx] == posc)
                if p.sum() < 5 or (~p).sum() < 5: continue
                kind = "chg" if src == "c" else "new"
                S = {c: f(ens(arm, c, kind)[idx, j]) for c in ("true", "shuf", "ident")}
                up = {q: np.where(pid[idx] == q)[0] for q in np.unique(pid[idx])}; keys = list(up); d_sh, a_tr, a_sh = [], [], []
                for _ in range(B):
                    ii = np.concatenate([up[k] for k in rng.choice(keys, len(keys))])
                    a1, a2 = au(p[ii], S["true"][ii]), au(p[ii], S["shuf"][ii]); d_sh.append(a1 - a2)
                d_sh = np.array(d_sh); d_sh = d_sh[~np.isnan(d_sh)]
                rows.append(dict(arm=arm, split=sp, unit=u, task=t, n_pos=int(p.sum()), n_neg=int((~p).sum()), auroc_true=au(p, S["true"]), auroc_shuf=au(p, S["shuf"]),
                                 auroc_ident=au(p, S["ident"]), delta_true_minus_shuf=au(p, S["true"]) - au(p, S["shuf"]), d_lo=np.percentile(d_sh, 2.5), d_hi=np.percentile(d_sh, 97.5),
                                 meanP_true=float(S["true"][p].mean()), meanP_shuf=float(S["shuf"][p].mean()), meanP_ident=float(S["ident"][p].mean())))
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(R[(R.unit == "pooled")][["arm", "split", "task", "n_pos", "n_neg", "auroc_true", "auroc_shuf", "auroc_ident", "delta_true_minus_shuf", "d_lo", "d_hi"]].round(3).to_string(index=False))
