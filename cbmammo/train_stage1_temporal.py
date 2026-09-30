"""Stage 1, temporal (paper 2): train the change/new-detection heads on top of a FROZEN
paper-1 base checkpoint. Mirrors train_stage1.py's CLI/config/checkpoint-dump conventions
closely (same --config/--seed/--out/--override, same run.json/_running lock, same dump-every-
split pattern) so the two are easy to run side by side; only the model/data/loss are swapped.

python -m cbmammo.train_stage1_temporal --config configs/cb_temporal_dinov2.yaml [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from . import data as D
from . import temporal_data as TD
from .model import build
from .temporal_concepts import TEMPORAL_HEADS, CHANGE_CLASSES, NEW_CLASSES, IGNORE as T_IGNORE, encode_change, encode_new
from .temporal_model import TemporalConceptModel, OpaqueTemporalModel, temporal_loss


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def device_of(cfg):
    if cfg.get("device"):
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_temporal(cfg, dev):
    """cfg['base'] is a normal stage-1 model cfg (encoder/model/arm), built with model.build()
    exactly like train_stage1.py does. cfg['base_checkpoint'], if given, loads those weights
    (state_dict, strict=False so the temporal-only submodules added later are unaffected).
    cfg['freeze_base'] (default True) freezes every base parameter -- the paired manifest is
    far smaller than paper 1's training set, so the default is to reuse the base representation
    unchanged and only train the new cross-attention + change/new heads (see
    paper2_brief_temporal_change.md section 11)."""
    base = build(cfg["base"]).to(dev)
    ckpt = cfg.get("base_checkpoint")
    if ckpt:
        sd = torch.load(ckpt, map_location=dev)
        sd = sd.get("state_dict", sd)
        missing, unexpected = base.load_state_dict(sd, strict=False)
        print(f"[temporal] loaded base checkpoint {ckpt}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    if cfg.get("freeze_base", True):
        for p in base.parameters():
            p.requires_grad = False
        base.eval()
    tm = cfg.get("model", {})
    Wrap = TemporalConceptModel if cfg["base"]["arm"] == "cb" else OpaqueTemporalModel
    return Wrap(base, tm.get("heads", 8), tm.get("dropout", 0.1), tm.get("prior_mode", "prior")).to(dev)


def temporal_class_weights(records, dev, power=0.5):
    """Per-(change|new)-head class weights, same power-mean convention as train_stage1's
    class_weights(). Uses temporal_concepts.encode_change/encode_new on every record so the
    weights reflect the SAME label definitions the loss trains on (incl. the increased+decreased
    merge and the weak-tag exclusion)."""
    change_counts = {h: np.zeros(len(CHANGE_CLASSES), int) for h in TEMPORAL_HEADS}
    new_counts = {h: np.zeros(len(NEW_CLASSES), int) for h in TEMPORAL_HEADS}
    for r in records:
        t = r["embed_temporal"]
        cl = encode_change(t["change"])
        nl, nw = encode_new(t["change"], t["new_candidate"])
        for h in TEMPORAL_HEADS:
            if cl[h] != T_IGNORE:
                change_counts[h][cl[h]] += 1
            if nl[h] != T_IGNORE and nw[h] > 0:
                new_counts[h][nl[h]] += 1
    def to_w(counts):
        out = {}
        for h, c in counts.items():
            c = np.maximum(c, 1).astype(float)
            w = (c.sum() / c) ** power
            out[h] = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
        return out
    return to_w(change_counts), to_w(new_counts)


def is_labeled(r, include_weak=False):
    """True if this pair can contribute to ANY change/new loss or metric term."""
    t = r["embed_temporal"]
    cl = encode_change(t["change"]); nl, nw = encode_new(t["change"], t["new_candidate"], include_weak=include_weak)
    return any(cl[h] != T_IGNORE for h in TEMPORAL_HEADS) or any(nl[h] != T_IGNORE and nw[h] > 0 for h in TEMPORAL_HEADS)


def eval_subset(rs, n_unlabeled, seed, include_weak=False):
    """All labelled pairs (the only ones any F1/AUROC counts) + a fixed random handful of unlabelled
    ones, kept solely so the prediction-distribution diagnostic can still look at unlabelled inputs.
    Replaces the earlier `rs[:max_test]` prefix cap, which silently dropped most labelled test pairs."""
    lab = [r for r in rs if is_labeled(r, include_weak)]
    unl = [r for r in rs if not is_labeled(r, include_weak)]
    random.Random(seed).shuffle(unl)
    return lab + unl[:n_unlabeled]


def auroc_binary(y, s):
    """Rank-based AUROC (Mann-Whitney, ties = 0.5); None if a class is missing."""
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def macro_auroc_score(pred, min_n=10):
    """Mean over eligible heads of the one-vs-rest macro AUROC (classes present with both sides).
    Threshold-free and prior-free: a collapsed/constant predictor scores 0.5, unlike argmax F1,
    which rewards predicting the majority class."""
    scores = {}
    for j, h in enumerate(TEMPORAL_HEADS):
        for kind in ("change", "new"):
            y = pred[f"{kind}_labels"][:, j].numpy(); P = pred[f"p_{kind}_{h}"].numpy()
            m = y != T_IGNORE
            if kind == "new":
                m &= pred["new_weights"][:, j].numpy() > 0
            if m.sum() < min_n or len(np.unique(y[m])) < 2:
                continue
            a = [auroc_binary(y[m] == k, P[m][:, k]) for k in np.unique(y[m])]
            a = [x for x in a if x is not None]
            if a:
                scores[f"{kind}_{h}"] = float(np.mean(a))
    return (float(np.mean(list(scores.values()))) if scores else 0.5), scores


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval()
    res = {"idx": [], "change_labels": [], "new_labels": [], "new_weights": [],
           **{f"p_change_{h}": [] for h in TEMPORAL_HEADS}, **{f"p_new_{h}": [] for h in TEMPORAL_HEADS}}
    for b in loader:
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda", dtype=torch.bfloat16):
            o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev),
                       b["prior_cc"].to(dev), b["prior_mlo"].to(dev), b["prior_view_mask"].to(dev))
        for h in TEMPORAL_HEADS:
            res[f"p_change_{h}"].append(o["change_logits"][h].float().softmax(-1).cpu())
            res[f"p_new_{h}"].append(o["new_logits"][h].float().softmax(-1).cpu())
        res["idx"].append(b["idx"]); res["change_labels"].append(b["change_labels"])
        res["new_labels"].append(b["new_labels"]); res["new_weights"].append(b["new_weights"])
    return {k: torch.cat(v) for k, v in res.items()}


def macro_f1_score(pred, min_n=10):
    """Mean macro-F1 over change/new heads that have >=min_n labelled, >1-class examples --
    same selection criterion as train_stage1.val_score, just over the temporal heads."""
    from .metrics import macro_f1
    scores = {}
    for h in TEMPORAL_HEADS:
        y = pred["change_labels"][:, TEMPORAL_HEADS.index(h)].numpy()
        m = y != T_IGNORE
        if m.sum() >= min_n and len(np.unique(y[m])) > 1:
            scores[f"change_{h}"] = macro_f1(y[m], pred[f"p_change_{h}"][m].argmax(1).numpy())
        yn, wn = pred["new_labels"][:, TEMPORAL_HEADS.index(h)].numpy(), pred["new_weights"][:, TEMPORAL_HEADS.index(h)].numpy()
        m2 = (yn != T_IGNORE) & (wn > 0)
        if m2.sum() >= min_n and len(np.unique(yn[m2])) > 1:
            scores[f"new_{h}"] = macro_f1(yn[m2], pred[f"p_new_{h}"][m2].argmax(1).numpy())
    return (float(np.mean(list(scores.values()))) if scores else 0.0), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--override", nargs="*", default=[], help="dotted key=value, e.g. train.epochs=2")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.override:
        k, v = kv.split("=", 1)
        node = cfg
        *path, last = k.split(".")
        for p in path:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(v)
    seed = a.seed if a.seed is not None else cfg.get("seed", 0)
    out = a.out or os.path.join(cfg.get("out_root", "runs"), f"{cfg['name']}_s{seed}")
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "run.json")) or os.path.exists(os.path.join(out, "_running")):
        print(f"[skip] {out}: already finished or running in another process"); return
    open(os.path.join(out, "_running"), "w").write(str(os.getpid()))
    set_seed(seed)
    dev = device_of(cfg)
    tr, dc = cfg["train"], cfg["data"]
    amp = tr.get("amp", True)

    recs = D.load_manifest(dc["train_manifests"])
    train_r = [r for r in recs if r["split"] == "train"]
    val_r = [r for r in recs if r["split"] == "val"]
    if tr.get("max_train"):
        random.Random(seed).shuffle(train_r); train_r = train_r[: tr["max_train"]]
    if tr.get("max_val"):
        val_r = val_r[: tr["max_val"]]
    size = tuple(dc.get("size", [1024, 640]))
    include_weak = bool(tr.get("include_weak_new", False))
    if tr.get("eval_labeled_only"):
        n_unl = tr.get("eval_unlabeled", 200)
        val_r = eval_subset(val_r, n_unl, seed, include_weak)
    sampler = None
    if tr.get("label_aware_sampler"):
        lab = np.array([is_labeled(r, include_weak) for r in train_r])
        frac = tr.get("labeled_frac", 0.5)
        w = np.where(lab, frac / max(1, lab.sum()), (1 - frac) / max(1, (~lab).sum()))
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=tr.get("epoch_samples", 3000),
                                        replacement=True, generator=torch.Generator().manual_seed(seed))
        print(f"[sampler] label-aware: {int(lab.sum())}/{len(lab)} labelled train pairs, labeled_frac={frac}, "
              f"epoch_samples={tr.get('epoch_samples', 3000)}", flush=True)
    mk = lambda rs, train, smp=None: DataLoader(
        TD.TemporalBreastDataset(rs, size, train, dc.get("clahe", True), dc.get("cache_dir"), include_weak_new=include_weak),
        batch_size=tr["batch_size"], shuffle=(train and smp is None), sampler=smp, num_workers=tr.get("workers", 4),
        collate_fn=TD.collate, drop_last=train, pin_memory=dev.type == "cuda")
    tl, vl = mk(train_r, True, sampler), mk(val_r, False)
    print(f"train {len(train_r)}  val {len(val_r)}  device {dev}  include_weak_new={include_weak}")

    model = build_temporal(cfg, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M "
          f"(base frozen: {not any(p.requires_grad for p in model.base.parameters())})")
    base_lr = tr.get("lr", 3e-4)
    opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=tr.get("wd", 0.05))
    steps = tr["epochs"] * max(1, len(tl))
    warm = int(0.05 * steps)

    # Resume support: a prior attempt in this SAME `out` dir may have been cut off by the
    # ssh run-clock timeout (exit 124) partway through -- if a completed-epoch log + a best.pt
    # from it are on disk (and no run.json, i.e. the run never reached its final dump), pick up
    # training from the next epoch instead of re-training from epoch 0 (each epoch here costs
    # ~2.5-3h, so restarting cold would silently waste every already-completed epoch's compute).
    # Optimizer/scheduler momentum is NOT preserved across a resume -- only model weights and
    # the epoch/best/bad bookkeeping -- a known, accepted imprecision of this quick resume path.
    start_ep, best, bad, log = 0, -1, 0, []
    log_path, best_path = os.path.join(out, "train_log.json"), os.path.join(out, "best.pt")
    if os.path.exists(log_path) and os.path.exists(best_path) and not os.path.exists(os.path.join(out, "run.json")):
        log = json.load(open(log_path))
        if log:
            sd = torch.load(best_path, map_location=dev)
            model.load_state_dict(sd, strict=False)
            best_idx = max(range(len(log)), key=lambda i: log[i]["val_score"])
            best, bad, start_ep = log[best_idx]["val_score"], len(log) - 1 - best_idx, len(log)
            print(f"[resume] {len(log)} epoch(s) already logged in {out}; resuming at epoch {start_ep} "
                  f"(best={best:.4f} @ epoch {best_idx}, bad={bad}); optimizer state re-initialized")

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps)))))
    for _ in range(start_ep * max(1, len(tl))):    # fast-forward the LR schedule on resume (see above);
        sched.step()                               # LambdaLR(last_epoch=N) needs 'initial_lr' set, which a
                                                     # freshly-built optimizer doesn't have -- step() N times instead
    change_cw, new_cw = temporal_class_weights(train_r, dev, tr.get("class_weight_power", 0.5))
    aux_lambda = cfg["model"].get("aux_lambda", 1.0) if cfg["base"]["arm"] == "opaque" else 1.0

    for ep in range(start_ep, tr["epochs"]):
        model.train(); t0 = time.time(); tot = 0
        for i, b in enumerate(tl):
            with torch.autocast(dev.type, enabled=amp and dev.type == "cuda", dtype=torch.bfloat16):
                o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev),
                          b["prior_cc"].to(dev), b["prior_mlo"].to(dev), b["prior_view_mask"].to(dev))
            loss, parts = temporal_loss(o, b["change_labels"].to(dev), b["new_labels"].to(dev),
                                       b["new_weights"].to(dev), aux_lambda=aux_lambda,
                                       change_cw=change_cw, new_cw=new_cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); tot += float(loss)
            if i % tr.get("log_every", 50) == 0:
                print(f"ep {ep} it {i}/{len(tl)} loss {float(loss):.4f}", flush=True)
        pv = predict(model, vl, dev, amp)
        f1_mean, per = macro_f1_score(pv)
        auc_mean, per_auc = macro_auroc_score(pv)
        score = auc_mean if tr.get("select_metric", "f1") == "auroc" else f1_mean
        log.append({"epoch": ep, "train_loss": tot / max(1, len(tl)), "val_score": score,
                    "val_f1": per, "val_f1_mean": f1_mean, "val_auroc": per_auc, "val_auroc_mean": auc_mean,
                    "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        # Written after EVERY epoch (not just at the end) so a resume after a run-clock kill
        # sees the true up-to-date best/bad state -- see the [resume] block above, which reads
        # this same file. Previously this was written once at loop-exit only, so a run killed
        # mid-loop left train_log.json holding whatever a PRIOR attempt in this `out` dir had
        # written (e.g. a stale pre-class-weight-fix run's log) -- a resume would then silently
        # pick up the wrong best-score threshold from that stale run instead of this one's.
        json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
        if score > best:
            best, bad = score, 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= tr.get("patience", 5):
                break

    sd = torch.load(os.path.join(out, "best.pt"), map_location=dev)
    model.load_state_dict(sd, strict=False)
    eval_sets = {"val": val_r}
    for mpath in dc.get("train_manifests", []) + dc.get("test_manifests", []):
        for r in D.load_manifest(mpath):
            if r["split"] == "test" or r["split"].startswith("test_"):
                eval_sets.setdefault(f"test_{r['dataset']}" if r["split"] == "test" else r["split"], []).append(r)
    max_test = tr.get("max_test")           # cap each eval split's size (smoke tests only; None = full, matching train_stage1.py)
    for name, rs in eval_sets.items():
        if not rs:
            continue
        if name != "val":
            if tr.get("eval_labeled_only"):
                rs = eval_subset(rs, tr.get("eval_unlabeled", 200), seed, include_weak)
            elif max_test:
                rs = rs[:max_test]
        pr = predict(model, mk(rs, False), dev, amp)
        pr["sample_id"] = [rs[i]["sample_id"] for i in pr["idx"].tolist()]
        torch.save(pr, os.path.join(out, f"dump_{name}.pt"))
        s, per = macro_f1_score(pr); au, per_au = macro_auroc_score(pr)
        print(f"[dump] {name}: n={len(rs)} mean-macroF1={s:.3f} {json.dumps({k: round(v, 3) for k, v in per.items()})}")
        print(f"[dump] {name}: mean-AUROC={au:.3f} {json.dumps({k: round(v, 3) for k, v in per_au.items()})}", flush=True)
    json.dump({"config": cfg, "seed": seed, "best_val": best}, open(os.path.join(out, "run.json"), "w"), indent=1)
    if os.path.exists(os.path.join(out, "_running")):
        os.remove(os.path.join(out, "_running"))


if __name__ == "__main__":
    main()
