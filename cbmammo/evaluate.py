"""Build every results table from stage-1 dumps and stage-2 generations.

python -m cbmammo.evaluate --runs runs/cb_dinov2_s0 runs/opq_dinov2_s0 ... --out results/
  [--compare runs/cb_dinov2_s0:runs/opq_dinov2_s0]

Tables (CSV):
  concepts.csv       per run x test set x head: macro-F1 [95% CI], AUC, n   (patient-clustered bootstrap)
  linear_probe.csv   logistic regression on frozen pooled features (lower bound)
  subgroups.csv      malignancy / BI-RADS by breast density and by population
  reports.csv        slot F1 + coverage for generated and template reports
  extractor.csv      instrument validation: extractor on real reports vs structured labels (CDD-CESM)
  intervention.csv   adoption / extractability / collateral change / BI-RADS coherence per head
  compare.csv        paired differences between two runs (e.g. cb - opaque), with p and equivalence
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import data as D
from .concepts import CONCEPT_HEADS, HEAD_BY_NAME, HEADS, IGNORE
from .metrics import cluster_bootstrap, macro_f1, paired_bootstrap_diff, safe_auc, tost_equivalence

NB = 1000


def _load(run, name):
    return torch.load(os.path.join(run, f"dump_{name}.pt"), map_location="cpu")


def _patients(ids, recs):
    return np.array([recs[i]["patient_id"] if i in recs else i for i in ids])


def run_label(run):
    rj = json.load(open(os.path.join(run, "run.json")))
    return f"{rj['config']['name']}_s{rj['seed']}"


def concept_table(runs, recs, nb=NB):
    rows = []
    for run in runs:
        lab = run_label(run)
        for p in sorted(glob.glob(os.path.join(run, "dump_test_*.pt"))):
            name = os.path.basename(p)[5:-3]
            d = torch.load(p, map_location="cpu")
            g = _patients(d["sample_id"], recs)
            for j, h in enumerate(HEADS):
                y = d["labels"][:, j].numpy(); m = y != IGNORE
                if m.sum() < 10 or len(np.unique(y[m])) < 2:
                    continue
                pr = d[f"p_{h.name}"].numpy()
                f1 = cluster_bootstrap(macro_f1, y[m], pr[m].argmax(1), g[m], nb)
                auc = cluster_bootstrap(safe_auc, y[m], pr[m], g[m], nb)
                rows.append({"run": lab, "set": name, "head": h.name, "n": int(m.sum()),
                             "f1": f1[0], "f1_lo": f1[1], "f1_hi": f1[2], "auc": auc[0], "auc_lo": auc[1], "auc_hi": auc[2]})
    return pd.DataFrame(rows)


def linear_probe(run, recs):
    tr = _load(run, "train")
    X = tr["feats"].numpy(); sc = StandardScaler().fit(X)
    rows = []
    for p in sorted(glob.glob(os.path.join(run, "dump_test_*.pt"))):
        d = torch.load(p, map_location="cpu"); name = os.path.basename(p)[5:-3]
        Xt = sc.transform(d["feats"].numpy())
        for j, h in enumerate(HEADS):
            y, yt = tr["labels"][:, j].numpy(), d["labels"][:, j].numpy()
            m, mt = y != IGNORE, yt != IGNORE
            if m.sum() < 20 or mt.sum() < 10 or len(np.unique(y[m])) < 2 or len(np.unique(yt[mt])) < 2:
                continue
            clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(sc.transform(X[m]), y[m])
            prob = np.zeros((mt.sum(), h.n)); prob[:, clf.classes_] = clf.predict_proba(Xt[mt])
            rows.append({"run": run_label(run), "set": name, "head": h.name, "n": int(mt.sum()),
                         "f1": macro_f1(yt[mt], prob.argmax(1)), "auc": safe_auc(yt[mt], prob)})
    return pd.DataFrame(rows)


def subgroup_table(runs, recs, nb=NB):
    rows = []
    for run in runs:
        for p in sorted(glob.glob(os.path.join(run, "dump_test_*.pt"))):
            d = torch.load(p, map_location="cpu"); name = os.path.basename(p)[5:-3]
            dens = np.array([recs[i]["values"].get("density") or "NA" for i in d["sample_id"]])
            g = _patients(d["sample_id"], recs)
            for head in ("malignant", "birads"):
                j = [h.name for h in HEADS].index(head)
                y = d["labels"][:, j].numpy(); pr = d[f"p_{head}"].numpy()
                for grp in ("A", "B", "C", "D"):
                    m = (y != IGNORE) & (dens == grp)
                    if m.sum() < 20 or len(np.unique(y[m])) < 2:
                        continue
                    fn = safe_auc if head == "malignant" else macro_f1
                    val = cluster_bootstrap(fn, y[m], pr[m] if head == "malignant" else pr[m].argmax(1), g[m], nb)
                    rows.append({"run": run_label(run), "set": name, "head": head, "density": grp, "n": int(m.sum()),
                                 "metric": "auc" if head == "malignant" else "f1", "value": val[0], "lo": val[1], "hi": val[2]})
    return pd.DataFrame(rows)


def _slot_rows(gen, key, lab, name):
    rows = []
    for h in HEADS:
        if h.name == "malignant":
            continue
        y, p, y_all, p_all = [], [], [], []
        for r in gen:
            gt = r["gt_values"].get(h.name)
            if gt is None or r.get(key) is None:
                continue
            s = r[key].get(h.name)
            y_all.append(h.classes.index(gt))
            p_all.append(h.classes.index(s) if s is not None else -1)     # missing slot = error
            if s is not None:
                y.append(y_all[-1]); p.append(p_all[-1])
        if len(y_all) >= 10 and len(set(y_all)) > 1:
            rows.append({"run": lab, "set": name, "source": key.replace("slots_", ""), "head": h.name,
                         "n": len(y_all), "coverage": float(np.mean(np.array(p_all) >= 0)),
                         "f1_on_extracted": _f1(y, p, h.n) if y else np.nan,
                         "f1_strict": _f1(y_all, p_all, h.n)})       # headline number
    return rows


def _f1(y, p, n):
    from sklearn.metrics import f1_score
    return f1_score(y, p, labels=list(range(n)), average="macro", zero_division=0)


def report_tables(stage2_dirs):
    rep, ext = [], []
    for s2 in stage2_dirs:
        rj = json.load(open(os.path.join(s2, "run.json")))
        lab = f"{run_label(rj['stage1'])}/{rj['config']['name']}"
        for p in sorted(glob.glob(os.path.join(s2, "gen_*.jsonl"))):
            name = os.path.basename(p)[4:-6]
            gen = [json.loads(l) for l in open(p)]
            rep += _slot_rows(gen, "slots_generated", lab, name)
            if gen and gen[0].get("slots_template") is not None:
                rep += _slot_rows(gen, "slots_template", lab, name)
            if any(r.get("slots_real_report") for r in gen):
                ext += _slot_rows([r for r in gen if r.get("slots_real_report")], "slots_real_report", lab, name)
    return pd.DataFrame(rep), pd.DataFrame(ext).drop_duplicates(subset=["set", "head"]) if ext else pd.DataFrame()


SUSPICIOUS = {("mass_shape", "irregular"), ("mass_margin", "spiculated"), ("mass_margin", "indistinct"),
              ("mass_margin", "microlobulated"), ("calc_morphology", "fine_linear_branching"),
              ("calc_morphology", "fine_pleomorphic"), ("calc_distribution", "segmental"),
              ("calc_distribution", "linear"), ("distortion", "present")}


def intervention_table(stage2_dirs):
    rows = []
    for s2 in stage2_dirs:
        for p in glob.glob(os.path.join(s2, "intervene_*.jsonl")):
            it = pd.DataFrame([json.loads(l) for l in open(p)])
            if it.empty:
                continue
            for (head, forced), g in it.groupby(["head", "forced"]):
                others = [h.name for h in CONCEPT_HEADS if h.name != head and h.name != HEAD_BY_NAME[head].parent]
                collateral = np.mean([any(r["before"].get(o) != r["after"].get(o) for o in others) for _, r in g.iterrows()])
                coh = np.nan
                if (head, forced) in SUSPICIOUS:
                    b = [r["after"].get("birads") for _, r in g.iterrows() if r["after"].get("birads")]
                    coh = float(np.mean([int(x) >= 4 for x in b])) if b else np.nan
                rows.append({"stage2": s2, "head": head, "forced": forced, "n": len(g),
                             "adoption": g["adopted"].mean(), "extractable": g["extractable"].mean(),
                             "collateral_change": collateral, "birads_ge4_after_suspicious": coh})
    return pd.DataFrame(rows)


def compare(run_a, run_b, recs, margin=0.03, nb=NB):
    rows = []
    for p in sorted(glob.glob(os.path.join(run_a, "dump_test_*.pt"))):
        name = os.path.basename(p)[5:-3]
        pb = os.path.join(run_b, os.path.basename(p))
        if not os.path.exists(pb):
            continue
        A, B = torch.load(p, map_location="cpu"), torch.load(pb, map_location="cpu")
        assert A["sample_id"] == B["sample_id"], "runs must share the test set order"
        g = _patients(A["sample_id"], recs)
        for j, h in enumerate(HEADS):
            y = A["labels"][:, j].numpy(); m = y != IGNORE
            if m.sum() < 20 or len(np.unique(y[m])) < 2:
                continue
            d, lo, hi, pv = paired_bootstrap_diff(macro_f1, y[m], A[f"p_{h.name}"][m].argmax(1).numpy(),
                                                  B[f"p_{h.name}"][m].argmax(1).numpy(), g[m], nb)
            row = {"a": run_label(run_a), "b": run_label(run_b), "set": name, "head": h.name, "n": int(m.sum()),
                   "f1_diff": d, "lo": lo, "hi": hi, "p": pv, f"equivalent_at_{margin}": tost_equivalence(lo, hi, margin)}
            if h.n == 2:
                da, la, ha, pa_ = paired_bootstrap_diff(safe_auc, y[m], A[f"p_{h.name}"][m][:, 1].numpy(),
                                                        B[f"p_{h.name}"][m][:, 1].numpy(), g[m], nb)
                row.update(auc_diff=da, auc_lo=la, auc_hi=ha, auc_p=pa_)
            rows.append(row)
    return pd.DataFrame(rows)


def seed_summary(ct: pd.DataFrame):
    """Mean +- SD across seeds for each (config, set, head)."""
    if ct.empty:
        return ct
    ct = ct.copy(); ct["config"] = ct["run"].str.replace(r"_s\d+$", "", regex=True)
    return ct.groupby(["config", "set", "head"]).agg(n_seeds=("f1", "size"), f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
                                                     auc_mean=("auc", "mean"), auc_sd=("auc", "std")).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare", nargs="*", default=[], help="runA:runB pairs")
    ap.add_argument("--bootstrap", type=int, default=NB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    manifests = sorted({m for r in a.runs for m in (lambda c: c["data"]["train_manifests"] + c["data"].get("test_manifests", []))(
        json.load(open(os.path.join(r, "run.json")))["config"])})
    recs = {r["sample_id"]: r for r in D.load_manifest(manifests)}
    ct = concept_table(a.runs, recs, a.bootstrap); ct.to_csv(os.path.join(a.out, "concepts.csv"), index=False)
    seed_summary(ct).to_csv(os.path.join(a.out, "concepts_by_seed.csv"), index=False)
    pd.concat([linear_probe(r, recs) for r in a.runs if os.path.exists(os.path.join(r, "dump_train.pt"))]
              or [pd.DataFrame()]).to_csv(os.path.join(a.out, "linear_probe.csv"), index=False)
    subgroup_table(a.runs, recs, a.bootstrap).to_csv(os.path.join(a.out, "subgroups.csv"), index=False)
    s2 = [d for r in a.runs for d in glob.glob(os.path.join(r, "stage2_*")) if os.path.exists(os.path.join(d, "run.json"))]
    if s2:
        rep, ext = report_tables(s2)
        rep.to_csv(os.path.join(a.out, "reports.csv"), index=False); ext.to_csv(os.path.join(a.out, "extractor.csv"), index=False)
        intervention_table(s2).to_csv(os.path.join(a.out, "intervention.csv"), index=False)
    if a.compare:
        pd.concat([compare(*pair.split(":"), recs, nb=a.bootstrap) for pair in a.compare]).to_csv(
            os.path.join(a.out, "compare.csv"), index=False)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
