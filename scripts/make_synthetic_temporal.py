"""Paired current+prior synthetic dataset for paper 2's CPU smoke test (extends
scripts/make_synthetic.py; NOT for results). Reuses draw() from make_synthetic.py unchanged
and adds a controlled synthetic 'change' between two draws of the same synthetic breast, so
every temporal class (stable/increased/decreased/resolved) and the new-detection heads have
guaranteed-nonzero examples to exercise the training loop end-to-end.
"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from make_synthetic import draw
from cbmammo import concepts as C
from cbmammo.temporal_concepts import PRESENCE_HEADS

CHANGE_KINDS = ("stable", "increased", "decreased", "resolved", "new")


def _values(mass_present, shape="oval", margin="circumscribed"):
    v = {"density": "B", "mass": "present" if mass_present else "absent",
         "calc": "absent", "asymmetry": "absent", "distortion": "absent"}
    if mass_present:
        v["mass_shape"], v["mass_margin"] = shape, margin
    v["birads"] = "4" if (mass_present and margin == "spiculated") else ("2" if mass_present else "1")
    v["malignant"] = "benign"
    return v


def main(out="smoke_temporal", n=40, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(f"{out}/img", exist_ok=True)
    os.makedirs(f"{out}/manifests", exist_ok=True)
    recs = []
    for i in range(n):
        kind = CHANGE_KINDS[i % len(CHANGE_KINDS)]
        split = "val" if i % 4 == 0 else "train"   # NOT i % 5: kind is i % 5, coupling the two moduli
                                                    # made every val row 'stable' (i%5==0 -> kind index 0)

        if kind == "new":
            prior_v, cur_v = _values(False), _values(True, "irregular", "spiculated")
        elif kind == "resolved":
            prior_v, cur_v = _values(True, "oval", "circumscribed"), _values(False)
        elif kind == "increased":
            prior_v, cur_v = _values(True, "oval", "circumscribed"), _values(True, "irregular", "spiculated")
        elif kind == "decreased":
            prior_v, cur_v = _values(True, "irregular", "spiculated"), _values(True, "oval", "circumscribed")
        else:  # stable
            prior_v = cur_v = _values(bool(i % 2), "oval", "circumscribed")

        views, p_views = {}, {}
        for vp in ("CC", "MLO"):
            p1 = f"{out}/img/{i}_{vp}_cur.png"; cv2.imwrite(p1, (draw(cur_v, rng) * 255).astype(np.uint8)); views[vp] = p1
            p2 = f"{out}/img/{i}_{vp}_prior.png"; cv2.imwrite(p2, (draw(prior_v, rng) * 255).astype(np.uint8)); p_views[vp] = p2

        change = {h: None for h in PRESENCE_HEADS}
        new_candidate = {h: None for h in PRESENCE_HEADS}
        if kind in ("stable", "increased", "decreased", "resolved"):
            change["mass"] = kind
        else:  # new
            new_candidate["mass"] = "strong"

        lab = C.encode(cur_v)
        recs.append({
            "sample_id": f"synthtemp:{i}", "dataset": "embed_temporal", "patient_id": f"p{i}",
            "study_id": str(i), "side": "L", "split": split, "views": views,
            "values": C.decode(lab), "labels": lab, "report": None, "population": "synthetic",
            "embed_temporal": {
                "prior_study_id": f"{i}_prior", "prior_views": p_views,
                "prior_values": C.decode(C.encode(prior_v)), "prior_labels": C.encode(prior_v),
                "change": change, "new_candidate": new_candidate,
            },
        })
    with open(f"{out}/manifests/embed_temporal_synthetic.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    print(f"synthetic temporal data written to {out}: {len(recs)} records, "
          f"kinds={ {k: sum(1 for i in range(n) if CHANGE_KINDS[i%len(CHANGE_KINDS)]==k) for k in CHANGE_KINDS} }")


if __name__ == "__main__":
    main()
