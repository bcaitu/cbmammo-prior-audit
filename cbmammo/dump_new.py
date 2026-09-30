"""Dump predictions of an already trained stage-1 run on a new manifest (no retraining).

    python -m cbmammo.dump_new --run runs/cb_dinov2_s0 --manifest manifests/mmc_astana.jsonl \
        --name test_mmc_astana --cache_dir ~/local_kz/cache_mmc

Writes <run or --out_dir>/dump_<name>.pt with the same fields as the dumps written at the end of training
(p_<head>, labels, cond, feats, idx, sample_id), so fewshot/oracle tools can read it."""
from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from . import data as D
from .model import build
from .train_stage1 import predict, val_score, device_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out_dir", default=None, help="write the dump here instead of the run directory")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.run, "run.json")))["config"]
    dev = device_of(cfg)
    amp = cfg.get("train", {}).get("amp", True)
    model = build(cfg).to(dev)
    sd = torch.load(os.path.join(a.run, "best.pt"), map_location=dev)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("enc.m.")]
    assert not bad and not unexpected, f"checkpoint mismatch: missing={bad[:5]} unexpected={list(unexpected)[:5]}"
    model.eval()
    recs = D.load_manifest(a.manifest, splits=["test"])
    dc = cfg["data"]
    dl = DataLoader(D.BreastDataset(recs, tuple(dc.get("size", [1024, 640])), False, dc.get("clahe", True), a.cache_dir),
                    batch_size=a.batch, shuffle=False, num_workers=a.workers, collate_fn=D.collate, pin_memory=dev.type == "cuda")
    pr = predict(model, dl, dev, amp)
    pr["sample_id"] = [recs[i]["sample_id"] for i in pr["idx"].tolist()]
    od = a.out_dir or a.run
    os.makedirs(od, exist_ok=True)
    torch.save(pr, os.path.join(od, f"dump_{a.name}.pt"))
    s, per = val_score(pr)
    print(json.dumps({"run": os.path.basename(a.run.rstrip("/")), "name": a.name, "n": len(recs), "macroF1_mean": round(s, 3)}))


if __name__ == "__main__":
    main()
