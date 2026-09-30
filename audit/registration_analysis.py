"""Evaluation of the registration / difference-image baseline (audit/registration_baseline.py output) against the coded change labels.
usage: python audit/registration_analysis.py <reg_out dir> <results csv> [B=1000]     (writes <results csv> and <results csv without .csv>_controls.csv)
AUROC per task with patient-cluster bootstrap intervals, for the held-out splits pooled, each split, and a split-stratified average
(weights n_pos*n_neg; pooling across splits is confounded because the changed rate differs between splits)."""
import json, sys, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings("ignore")
D, OUT = sys.argv[1], sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
meta = json.load(open(f"{D}/meta.json")); Z = np.load(f"{D}/scores.npz"); K = meta["keys"]; PH = meta["presence_heads"]
def breast(x):                                            # max over the available views
    with np.errstate(all="ignore"): return np.nanmax(x, axis=1)
S = breast(Z["true"]); ki = {k: i for i, k in enumerate(K)}
cl = np.array(meta["change_labels"]); nl = np.array(meta["new_labels"]); nw = np.array(meta["new_weights"])
split = np.array(meta["split"]); pid = np.array(meta["patient_id"])
def pooled_change(r): v = set(x for x in r if x >= 0); return 2 if 2 in v else 1 if 1 in v else 0 if 0 in v else -1
def pooled_new(a, b): return 1 if any(l == 1 and w > 0 for l, w in zip(a, b)) else (0 if any(l == 0 for l in a) else -1)
Y = {h: cl[:, j] for j, h in enumerate(PH)}; Y["pooled"] = np.array([pooled_change(r) for r in cl])
N = {h: np.where(nw[:, j] > 0, nl[:, j], -1) for j, h in enumerate(PH)}; N["pooled"] = np.array([pooled_new(a, b) for a, b in zip(nl, nw)])
def auroc(y, s):
    y = np.asarray(y, bool); ok = ~np.isnan(s); y, s = y[ok], s[ok]; n1 = y.sum(); n0 = len(y) - n1
    return np.nan if n1 == 0 or n0 == 0 else float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def strat(y, s, sp):
    num = den = 0.0
    for k in np.unique(sp):
        m = sp == k; yy = np.asarray(y[m], bool); ok = ~np.isnan(s[m]); n1, n0 = yy[ok].sum(), (~yy[ok]).sum()
        a = auroc(y[m], s[m])
        if n1 and n0 and not np.isnan(a): num += n1 * n0 * a; den += n1 * n0
    return num / den if den else np.nan
TASKS = {  # name: (label source, keep, positive, score name)
    "changed_vs_stable": ("c", lambda y: np.isin(y, [0, 2]), lambda y: y == 2, "abs"),
    "resolved_vs_stable": ("c", lambda y: np.isin(y, [0, 1]), lambda y: y == 1, "neg"),
    "anychange_vs_stable": ("c", lambda y: y >= 0, lambda y: y > 0, "abs"),
    "new_vs_notnew(silver)": ("n", lambda y: y >= 0, lambda y: y == 1, "pos"),
}
GROUPS = {"heldout_pooled (split priors CONFOUND)": ["val", "test", "test_embed_ge", "test_embed_fuji"], "val": ["val"], "test": ["test"], "GE": ["test_embed_ge"], "fuji": ["test_embed_fuji"]}
rng = np.random.default_rng(0); rows = []
def add_row(group, unit, task, variant, y, s, sp, pids, stratified):
    n1, n0 = int(y.sum()), int((~y).sum()); row = dict(group=group, unit=unit, task=task, score=variant, n_pos=n1, n_neg=n0, n_scored=int((~np.isnan(s)).sum()))
    if n1 >= 1 and n0 >= 1:
        f = (lambda yy, ss, spp: strat(yy, ss, spp)) if stratified else (lambda yy, ss, spp: auroc(yy, ss))
        row["auroc"] = f(y, s, sp)
        if n1 >= 5 and n0 >= 5:
            up = {p: np.where(pids == p)[0] for p in np.unique(pids)}; keys = list(up); bs = []
            for _ in range(B):
                ii = np.concatenate([up[k] for k in rng.choice(keys, len(keys))]); bs.append(f(y[ii], s[ii], sp[ii]))
            bs = np.array(bs); bs = bs[~np.isnan(bs)]; row["ci_lo"], row["ci_hi"] = np.percentile(bs, [2.5, 97.5]); row["n_patients"] = len(keys)
    rows.append(row)
for variant in ("_q99", "_q999", "_mf"):                                                # 99th percentile, 99.9th percentile, lesion-scale maximum
    for tname, (src, keep, pos, sname) in TASKS.items():
        for u in ["pooled"] + PH:
            y_all = (Y if src == "c" else N)[u]
            for gname, sp_ in GROUPS.items():
                m = np.isin(split, sp_) & keep(y_all)
                if m.sum() == 0: continue
                add_row(gname, u, tname, sname + variant, pos(y_all[m]), S[m, ki[sname + variant]], split[m], pid[m], False)
            m = np.isin(split, ["val", "test", "test_embed_ge", "test_embed_fuji"]) & keep(y_all)          # split-stratified
            add_row("heldout_split_stratified", u, tname, sname + variant, pos(y_all[m]), S[m, ki[sname + variant]], split[m], pid[m], True)
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
# controls -------------------------------------------------------------------------------------------------------
ci = Z["ctrl_index"]; ct = breast(Z["ctrl_true"]); rc = []
def ctrl(name, s_pos, s_neg, key):
    a, b = s_pos[:, ki[key]], s_neg[:, ki[key]]; ok = ~(np.isnan(a) | np.isnan(b)); y = np.r_[np.ones(ok.sum(), bool), np.zeros(ok.sum(), bool)]; s = np.r_[a[ok], b[ok]]
    boot = []
    for _ in range(B):
        ii = rng.integers(0, ok.sum(), ok.sum()); boot.append(auroc(np.r_[np.ones(len(ii), bool), np.zeros(len(ii), bool)], np.r_[a[ok][ii], b[ok][ii]]))
    rc.append(dict(control=name, score=key, n_pairs=int(ok.sum()), auroc=auroc(y, s), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5)))
for st in ("q99", "q999", "mf"):
    ctrl(f"different-patient prior vs true prior (abs score)", breast(Z["ctrl_mismatched"]), ct, "abs_" + st)
    for k in [x for x in Z.files if x.startswith("blob_")]:
        ctrl(f"synthetic lesion, amplitude {k.split('_')[1]} (pos score) vs no lesion", breast(Z[k]), ct, "pos_" + st)
cdf = pd.DataFrame(rc); cdf.to_csv(OUT.replace(".csv", "_controls.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
cdf["stat"] = cdf["score"].str.split("_").str[1]
blob_auc = cdf[cdf.control.str.contains("amplitude 0.40")].set_index("stat")["auroc"]; primary = blob_auc.idxmax()      # chosen from the synthetic-lesion control only
json.dump(dict(primary_stat=primary, synthetic_lesion_auroc_amp040=blob_auc.to_dict()), open(OUT.replace(".csv", "_primary.json"), "w"))
print("primary statistic (max synthetic-lesion AUROC at amplitude 0.40):", primary, blob_auc.round(3).to_dict())
show = df[(df.unit == "pooled") & (df.score.str.endswith("_" + primary)) & df.group.isin(["heldout_pooled (split priors CONFOUND)", "heldout_split_stratified", "test", "GE", "fuji"])]
print(show[["group", "task", "score", "n_pos", "n_neg", "n_scored", "auroc", "ci_lo", "ci_hi"]].round(3).to_string(index=False)); print(cdf.round(3).to_string(index=False))
print("views scored:", float((~np.isnan(Z["true"][..., 0])).mean()), "| failure codes:", {int(k): int((Z["true"][..., ki["fail"]] == k).sum()) for k in np.unique(Z["true"][..., ki["fail"]][~np.isnan(Z["true"][..., ki["fail"]])])}, "| ECC converged (of scored views):", float(np.nanmean(Z["true"][..., ki["ecc_ok"]])), "| mean overlap:", float(np.nanmean(Z["true"][..., ki["overlap"]])))
