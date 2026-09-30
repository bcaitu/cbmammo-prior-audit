"""Label-density counts for embed_temporal.jsonl (run from the repository root).
Reports, for the TRAIN split, how many pairs carry any change/new label, how many carry a change label, the chance that a
batch of 12 contains no label under uniform sampling, and per-descriptor class counts."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # repo root (run scripts from the repo root)
from cbmammo import data as D
from cbmammo.temporal_concepts import PRESENCE_HEADS, encode_change, encode_new, IGNORE
recs = D.load_manifest(["manifests/embed_temporal.jsonl"])
tr = [r for r in recs if r["split"] == "train"]
print("train records", len(tr))
anyl = anych = anynew = 0
cc = {h: [0, 0, 0] for h in PRESENCE_HEADS}; nc = {h: [0, 0] for h in PRESENCE_HEADS}
for r in tr:
    t = r["embed_temporal"]
    cl = encode_change(t["change"]); nl, nw = encode_new(t["change"], t["new_candidate"])
    a = b = False
    for h in PRESENCE_HEADS:
        if cl[h] != IGNORE: cc[h][cl[h]] += 1; a = True
        if nl[h] != IGNORE and nw[h] > 0: nc[h][nl[h]] += 1; b = True
    anych += a; anynew += b; anyl += (a or b)
print("records with >=1 labeled change head:", anych, f"({anych/len(tr):.2%})")
print("records with >=1 labeled new head   :", anynew, f"({anynew/len(tr):.2%})")
print("records with any label              :", anyl, f"({anyl/len(tr):.2%})")
print("P(batch of 12 has zero labels)=", round((1 - anyl / len(tr)) ** 12, 3))
print("change class counts per head [stable,resolved,changed]:", cc)
print("new class counts per head [not_new,new]:", nc)
