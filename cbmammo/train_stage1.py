"""Stage 1: train fusion + bottleneck + heads on a frozen encoder, then dump
per-sample predictions / bottleneck vectors / pooled features for every split.

python -m cbmammo.train_stage1 --config configs/cb_dinov2.yaml [--seed 1] [--out runs/x]
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
from torch.utils.data import DataLoader

from . import data as D
from .concepts import HEADS, IGNORE
from .metrics import macro_f1
from .model import build, head_weights, masked_ce


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def device_of(cfg):
    if cfg.get("device"):
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights(records, dev, power=0.5):
    cw = {}
    for name, c in D.label_counts(records).items():
        c = np.maximum(c, 1).astype(float)
        w = (c.sum() / c) ** power
        cw[name] = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    return cw


@torch.no_grad()
def predict(model, loader, dev, amp):
    model.eval()
    res = {"idx": [], "labels": [], "cond": [], "feats": [], **{f"p_{h.name}": [] for h in HEADS}}
    for b in loader:
        with torch.autocast(dev.type, enabled=amp and dev.type == "cuda", dtype=torch.bfloat16):
            o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev))
        for h in HEADS:
            res[f"p_{h.name}"].append(o["logits"][h.name].float().softmax(-1).cpu())
        res["cond"].append(o["cond"].float().cpu()); res["feats"].append(o["feats"].float().cpu())
        res["labels"].append(b["labels"]); res["idx"].append(b["idx"])
    return {k: torch.cat(v) for k, v in res.items()}


def val_score(pred, datasets: list | None = None) -> tuple:
    """Mean macro-F1 over heads that have labels in the split (model selection criterion).

    `datasets`: dataset name per row, same order as `pred["idx"]`. When given, the score is
    computed separately per dataset and averaged with EQUAL weight across datasets, so a much
    larger validation set (e.g. EMBED alongside CBIS-DDSM/VinDr) cannot dominate checkpoint
    selection just by size. `datasets=None` reproduces the original pooled-over-all-rows score
    exactly (used by every pre-EMBED config, so their results are unaffected by this addition).
    """
    if datasets is None:
        scores = {}
        for j, h in enumerate(HEADS):
            y = pred["labels"][:, j].numpy()
            m = y != IGNORE
            if m.sum() >= 10 and len(np.unique(y[m])) > 1:
                scores[h.name] = macro_f1(y[m], pred[f"p_{h.name}"][m].argmax(1).numpy())
        return (float(np.mean(list(scores.values()))) if scores else 0.0), scores
    datasets = np.asarray(datasets)
    per_ds = {}
    for ds in sorted(set(datasets)):
        m_ds = datasets == ds
        sub = {"labels": pred["labels"][m_ds], **{f"p_{h.name}": pred[f"p_{h.name}"][m_ds] for h in HEADS}}
        s, _ = val_score(sub)
        per_ds[ds] = s
    return (float(np.mean(list(per_ds.values()))) if per_ds else 0.0), per_ds


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
    train_r_full = [r for r in recs if r["split"] == "train"]
    val_r = [r for r in recs if r["split"] == "val"]
    if tr.get("max_train"):
        random.Random(seed).shuffle(train_r_full); train_r_full = train_r_full[: tr["max_train"]]
    if tr.get("max_val"):
        val_r = val_r[: tr["max_val"]]
    embed_neg_ratio = tr.get("embed_neg_ratio")            # None = old behaviour (exact reproduction)
    dataset_balanced_val = bool(tr.get("dataset_balanced_val", False))
    size = tuple(dc.get("size", [1024, 640]))
    mk = lambda rs, train: DataLoader(D.BreastDataset(rs, size, train, dc.get("clahe", True), dc.get("cache_dir")),
                                      batch_size=tr["batch_size"], shuffle=train, num_workers=tr.get("workers", 4),
                                      collate_fn=D.collate, drop_last=train, pin_memory=dev.type == "cuda")
    train_r = D.epoch_sample(train_r_full, embed_neg_ratio, seed=seed)
    tl, vl = mk(train_r, True), mk(val_r, False)
    val_ds = [r["dataset"] for r in val_r]
    print(f"train {len(train_r_full)} (epoch sample {len(train_r)})  val {len(val_r)}  device {dev}"
          f"  embed_neg_ratio={embed_neg_ratio}  dataset_balanced_val={dataset_balanced_val}")

    model = build(cfg).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("enc.")]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("enc.")]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f} M (encoder {sum(p.numel() for p in enc_p)/1e6:.2f} M)")
    base_lr = tr.get("lr", 3e-4)
    groups = [{"params": oth_p, "lr": base_lr}] + ([{"params": enc_p, "lr": base_lr * tr.get("enc_lr_mult", 0.1)}] if enc_p else [])
    opt = torch.optim.AdamW(groups, lr=base_lr, weight_decay=tr.get("wd", 0.05))
    steps = tr["epochs"] * max(1, len(tl))
    warm = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, warm)) *
                                              0.5 * (1 + math.cos(math.pi * min(1, s / max(1, steps)))))
    hw = head_weights(cfg["arm"], cfg["model"].get("aux_lambda", 0.1), cfg["model"].get("target_weight", 1.0))
    cw = class_weights(train_r, dev, tr.get("class_weight_power", 0.5))

    best, bad, log = -1, 0, []
    for ep in range(tr["epochs"]):
        if embed_neg_ratio is not None:
            train_r = D.epoch_sample(train_r_full, embed_neg_ratio, seed=seed * 10007 + ep)
            tl = mk(train_r, True)
        model.train(); t0 = time.time(); tot = 0
        for i, b in enumerate(tl):
            with torch.autocast(dev.type, enabled=amp and dev.type == "cuda", dtype=torch.bfloat16):
                o = model(b["cc"].to(dev), b["mlo"].to(dev), b["view_mask"].to(dev))
            loss, parts = masked_ce(o["logits"], b["labels"].to(dev), hw, cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); tot += float(loss)
            if i % tr.get("log_every", 50) == 0:
                print(f"ep {ep} it {i}/{len(tl)} loss {float(loss):.4f}", flush=True)
        pv = predict(model, vl, dev, amp)
        score, per = val_score(pv, val_ds if dataset_balanced_val else None)
        log.append({"epoch": ep, "train_loss": tot / max(1, len(tl)), "n_train_epoch": len(train_r),
                    "val_score": score, "val_f1": per, "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if score > best:
            best, bad = score, 0
            torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("enc.m.") or
                        model.enc.trainable}, os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= tr.get("patience", 5):
                break
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)

    # ---------- dump every split of every manifest with the best checkpoint
    sd = torch.load(os.path.join(out, "best.pt"), map_location=dev)
    model.load_state_dict(sd, strict=False)
    eval_sets = {"val": val_r}
    for mpath in dc.get("train_manifests", []) + dc.get("test_manifests", []):
        for r in D.load_manifest(mpath):
            if r["split"] == "test":
                eval_sets.setdefault(f"test_{r['dataset']}", []).append(r)
            elif r["split"].startswith("test_"):        # e.g. test_embed_ge, test_embed_fuji
                eval_sets.setdefault(r["split"], []).append(r)
    if dc.get("dump_train", True):
        eval_sets["train"] = train_r_full
    for name, rs in eval_sets.items():
        pr = predict(model, mk(rs, False), dev, amp)
        pr["sample_id"] = [rs[i]["sample_id"] for i in pr["idx"].tolist()]
        torch.save(pr, os.path.join(out, f"dump_{name}.pt"))
        ds_here = [rs[i]["dataset"] for i in pr["idx"].tolist()]
        s, per = val_score(pr, ds_here if dataset_balanced_val and len(set(ds_here)) > 1 else None)
        print(f"[dump] {name}: n={len(rs)} mean-macroF1={s:.3f} {json.dumps({k: round(v, 3) for k, v in per.items()})}")
    json.dump({"config": cfg, "seed": seed, "best_val": best}, open(os.path.join(out, "run.json"), "w"), indent=1)
    if os.path.exists(os.path.join(out, "_running")):
        os.remove(os.path.join(out, "_running"))


if __name__ == "__main__":
    main()
