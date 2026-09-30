"""Paper 2 unified evaluation: concept-differencing vs learned CB vs learned opaque, same labelled pairs.
usage: python temporal_eval.py <cd_out dir> <dumps root> <out csv> [B]
All AUROCs use patient-cluster bootstrap; paired differences share the same resamples."""
import json, sys, glob, os, numpy as np, pandas as pd, torch, warnings
warnings.filterwarnings("ignore")
CD, DUMPS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; B = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
PH = ["mass", "calc", "asymmetry", "distortion"]; UNITS = PH + ["pooled"]
meta = json.load(open(f"{CD}/meta.json")); Pz = np.load(f"{CD}/probs.npz"); models = meta["models"]
sid = np.array(meta["sample_id"]); pid = np.array(meta["patient_id"]); split = np.array(meta["split"]); pos_of = {s: i for i, s in enumerate(sid)}
cl = np.array(meta["change_labels"]); nl = np.array(meta["new_labels"]); nw = np.array(meta["new_weights"]); cols = meta["columns"]; ci = {c: i for i, c in enumerate(cols)}
def pooled_change(r): v = set(x for x in r if x >= 0); return 2 if 2 in v else 1 if 1 in v else 0 if 0 in v else -1
def pooled_new(a, b): return 1 if any(l == 1 and w > 0 for l, w in zip(a, b)) else (0 if any(l == 0 for l in a) else -1)
Y = {h: cl[:, j] for j, h in enumerate(PH)}; Y["pooled"] = np.array([pooled_change(r) for r in cl])
N = {h: np.where(nw[:, j] > 0, nl[:, j], -1) for j, h in enumerate(PH)}; N["pooled"] = np.array([pooled_new(a, b) for a, b in zip(nl, nw)])
n = len(sid); NAN = np.full(n, np.nan)
TASKS = {"changed_vs_stable": ("c", [0, 2], 2), "resolved_vs_stable": ("c", [0, 1], 1), "anychange_vs_stable": ("c", [0, 1, 2], None), "new_vs_notnew(silver)": ("n", [0, 1], 1)}
def mask_pos(u, t):
    src, keep, pos = TASKS[t][0], TASKS[t][1], TASKS[t][2]; y = (Y if src == "c" else N)[u]
    m = np.isin(y, keep); p = (y > 0) if pos is None else (y == pos); return m, p
# ---- method scores: dict method -> unit -> task -> array(n) (nan = not evaluated)
groups_c = {h: [i for c, i in ci.items() if c.split(":")[0] in g] for h, g in {"mass": ["mass", "mass_shape", "mass_margin"], "calc": ["calc", "calc_morphology", "calc_distribution"], "asymmetry": ["asymmetry"], "distortion": ["distortion"]}.items()}
cur = np.mean([Pz[f"cur__{m}"] for m in models], 0); pri = np.mean([Pz[f"pri__{m}"] for m in models], 0)
def cdiff():
    o = {}
    for h in PH:
        pp = ci[f"{h}:present"]; chg = 0.5 * np.abs(cur[:, groups_c[h]] - pri[:, groups_c[h]]).sum(1)
        o[h] = {"changed_vs_stable": chg, "resolved_vs_stable": pri[:, pp] - cur[:, pp], "anychange_vs_stable": chg, "new_vs_notnew(silver)": cur[:, pp] - pri[:, pp]}
    o["pooled"] = {t: np.max([o[h][t] for h in PH], 0) for t in TASKS}
    return o
M = {"concept_diff (label-free)": cdiff()}
def learned(arm):
    seeds = sorted(glob.glob(f"{DUMPS}/{arm}_temporal_lab_s*")); acc = {}; per_seed = []
    for sd in seeds:
        d = {}
        for f in glob.glob(f"{sd}/dump_*.pt"):
            pr = torch.load(f, map_location="cpu")
            for k, s in enumerate(pr["sample_id"]):
                if s not in pos_of: continue                     # unlabelled filler rows
                i = pos_of[s]; d[i] = {u: (pr[f"p_change_{u}"][k].numpy(), pr[f"p_new_{u}"][k].numpy()) for u in UNITS}
                lab = pr["change_labels"][k].numpy()
                assert lab[4] == Y["pooled"][i] and all(lab[j] == Y[h][i] for j, h in enumerate(PH)), ("label mismatch", s)
        per_seed.append(d)
    def build(dicts):
        o = {u: {t: NAN.copy() for t in TASKS} for u in UNITS}
        common = set.intersection(*[set(x) for x in dicts]) if dicts else set()
        for i in common:
            for u in UNITS:
                pc = np.mean([x[i][u][0] for x in dicts], 0); pn = np.mean([x[i][u][1] for x in dicts], 0)
                o[u]["changed_vs_stable"][i] = pc[2]; o[u]["resolved_vs_stable"][i] = pc[1]; o[u]["anychange_vs_stable"][i] = 1 - pc[0]; o[u]["new_vs_notnew(silver)"][i] = pn[1]
        return o
    return build(per_seed), [build([x]) for x in per_seed], len(per_seed)
SEEDS = {}
for arm, name in (("cb", "CB learned (concept heads)"), ("opq", "opaque learned"), ("cb_self", "CB current-only"), ("opq_self", "opaque current-only")):
    e, ps, k = learned(arm)
    if k: M[f"{name} [{k}-seed ens]"] = e; SEEDS[f"{name} [{k}-seed ens]"] = ps
from scipy.stats import rankdata
def auroc(y, s):
    y = np.asarray(y, bool); n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
GROUPS = {"heldout_pooled (split priors CONFOUND)": ["test", "test_embed_ge", "test_embed_fuji"], "val": ["val"], "test": ["test"], "GE": ["test_embed_ge"], "fuji": ["test_embed_fuji"]}
rng = np.random.default_rng(0); rows = []; paired = []
learned_names = [m for m in M if m.startswith(("CB", "opaque"))]
for gname, sp in GROUPS.items():
    gm = np.isin(split, sp)
    for u in UNITS:
        for t in TASKS:
            m0, p0 = mask_pos(u, t); base = gm & m0
            for meth in M:
                s = M[meth][u][t]; ok = base & ~np.isnan(s)
                idx = np.where(ok)[0]; n1 = int(p0[idx].sum()); n0 = len(idx) - n1
                row = dict(group=gname, unit=u, task=t, method=meth, n_pos=n1, n_neg=n0, auroc=auroc(p0[idx], s[idx]) if n1 and n0 else np.nan)
                if meth in SEEDS and n1 and n0:
                    sa = [auroc(p0[idx], x[u][t][idx]) for x in SEEDS[meth]]; row["seed_min"], row["seed_max"] = np.nanmin(sa), np.nanmax(sa)
                rows.append(row)
            # paired bootstrap on common records across all methods
            common = base.copy()
            for meth in M: common &= ~np.isnan(M[meth][u][t])
            idx = np.where(common)[0]; y = p0[idx]
            if y.sum() >= 5 and (~y).sum() >= 5:
                up = {p: np.where(pid[idx] == p)[0] for p in np.unique(pid[idx])}; keys = list(up)
                bs = {meth: [] for meth in M}
                for _ in range(B):
                    ii = np.concatenate([up[k] for k in rng.choice(keys, len(keys))])
                    for meth in M: bs[meth].append(auroc(y[ii], M[meth][u][t][idx][ii]))
                bs = {k: np.array(v) for k, v in bs.items()}
                for r in rows[-len(M):]:
                    b = bs[r["method"]]; b = b[~np.isnan(b)]; r["ci_lo"], r["ci_hi"] = np.percentile(b, [2.5, 97.5]); r["n_common"] = len(idx); r["n_patients"] = len(keys)
                names = list(M)
                for a in range(len(names)):
                    for b_ in range(a + 1, len(names)):
                        dd = bs[names[a]] - bs[names[b_]]; dd = dd[~np.isnan(dd)]
                        paired.append(dict(group=gname, unit=u, task=t, a=names[a], b=names[b_], delta=float(auroc(y, M[names[a]][u][t][idx]) - auroc(y, M[names[b_]][u][t][idx])), ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5), n_common=len(idx)))

# ---------------- split-stratified AUROC (weights n_pos*n_neg per stratum), stratified patient-cluster bootstrap
STRATA = ["test", "test_embed_ge", "test_embed_fuji"]; SG = "STRATIFIED (test, GE, fuji)"
def strat_block(u, t):
    m0, p0 = mask_pos(u, t); names = list(M); common = m0.copy()
    for meth in names: common &= ~np.isnan(M[meth][u][t])
    st = []
    for sp in STRATA:
        idx = np.where(common & (split == sp))[0]; y = p0[idx]
        if y.sum() >= 1 and (~y).sum() >= 1:
            up = {p: np.where(pid[idx] == p)[0] for p in np.unique(pid[idx])}
            st.append((sp, idx, y, up, float(y.sum() * (~y).sum())))
    if not st: return
    W = np.array([x[4] for x in st]); W = W / W.sum()
    def est(resample):
        out = {meth: 0.0 for meth in names}
        for w, (sp, idx, y, up, _) in zip(W, st):
            if resample:
                keys = list(up); ii = np.concatenate([up[k] for k in rng.choice(keys, len(keys))])
            else:
                ii = np.arange(len(idx))
            for meth in names:
                a = auroc(y[ii], M[meth][u][t][idx][ii]); out[meth] += w * (0.5 if np.isnan(a) else a)
        return out
    point = est(False); bs = {meth: [] for meth in names}
    for _ in range(B):
        e = est(True)
        for meth in names: bs[meth].append(e[meth])
    n1 = int(sum(x[2].sum() for x in st)); n0 = int(sum((~x[2]).sum() for x in st))
    for meth in names:
        b = np.array(bs[meth]); row = dict(group=SG, unit=u, task=t, method=meth, n_pos=n1, n_neg=n0, auroc=point[meth],
                 ci_lo=np.percentile(b, 2.5), ci_hi=np.percentile(b, 97.5), n_common=sum(len(x[1]) for x in st), strata=",".join(x[0] for x in st))
        if meth in SEEDS:
            sa = []
            for x in SEEDS[meth]:
                v = 0.0
                for w, (sp, idx, y, up, _) in zip(W, st):
                    a = auroc(y, x[u][t][idx]); v += w * (0.5 if np.isnan(a) else a)
                sa.append(v)
            row["seed_min"], row["seed_max"] = min(sa), max(sa)
        rows.append(row)
    for a_ in range(len(names)):
        for b_ in range(a_ + 1, len(names)):
            dd = np.array(bs[names[a_]]) - np.array(bs[names[b_]])
            paired.append(dict(group=SG, unit=u, task=t, a=names[a_], b=names[b_], delta=point[names[a_]] - point[names[b_]],
                               ci_lo=np.percentile(dd, 2.5), ci_hi=np.percentile(dd, 97.5), n_common=sum(len(x[1]) for x in st)))
for u in UNITS:
    for t in TASKS: strat_block(u, t)

R = pd.DataFrame(rows); R.to_csv(OUT, index=False); pd.DataFrame(paired).to_csv(OUT.replace(".csv", "_paired.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 40)
g = R[(R.group.str.startswith("STRATIFIED")) & (R.unit.isin(["pooled", "asymmetry", "mass"]))]
print(g[["unit", "task", "method", "n_pos", "n_neg", "auroc", "ci_lo", "ci_hi", "seed_min", "seed_max"]].round(3).to_string(index=False))
