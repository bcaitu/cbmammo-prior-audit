"""python -m cbmammo.prepare <dataset> --root <path> --out manifests/<dataset>.jsonl [--kw k=v ...]

Also: python -m cbmammo.prepare count manifests/*.jsonl   -> descriptor feasibility table
"""
import argparse
import sys

import pandas as pd

from .builders import BUILDERS
from ..data import load_manifest, label_counts
from ..concepts import HEADS


def count(paths):
    rows = []
    for p in paths:
        recs = load_manifest(p)
        splits = sorted({r["split"] for r in recs} | {"train", "val", "test"})
        for split in splits:
            rs = [r for r in recs if r["split"] == split]
            if not rs:
                continue
            for name, c in label_counts(rs).items():
                h = next(x for x in HEADS if x.name == name)
                rows.append({"manifest": p, "split": split, "head": name, "labelled": int(c.sum()),
                             **{f"{cls}": int(v) for cls, v in zip(h.classes, c)}})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df.fillna("").to_string(index=False))
    return df


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "count":
        count(sys.argv[2:]); sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(BUILDERS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kw", nargs="*", default=[], help="extra builder kwargs k=v (e.g. --kw legend_csv=/path)")
    a = ap.parse_args()
    BUILDERS[a.dataset](a.root, a.out, **dict(kv.split("=", 1) for kv in a.kw))
