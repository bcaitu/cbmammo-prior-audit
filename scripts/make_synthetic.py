"""Tiny synthetic mammography-like dataset for the CPU smoke test (NOT for results).
Masses are drawn as blobs whose shape/margin follow the label; calcs as dots."""
import json, os, random, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbmammo import concepts as C

def draw(v, rng, H=256, W=192):
    a = np.zeros((H, W), np.float32)
    cv2.ellipse(a, (0, H // 2), (int(W * .9), int(H * .45)), 0, -90, 90, 0.35 + 0.1 * "ABCD".index(v["density"]), -1)
    a += rng.normal(0, 0.03, a.shape).astype(np.float32)
    if v.get("mass") == "present":
        c = (rng.integers(40, W - 40), rng.integers(60, H - 60)); r = 14
        if v["mass_shape"] == "irregular":
            pts = np.array([[c[0] + r * np.cos(t) * rng.uniform(.5, 1.5), c[1] + r * np.sin(t) * rng.uniform(.5, 1.5)]
                            for t in np.linspace(0, 2 * np.pi, 9)], np.int32); cv2.fillPoly(a, [pts], 0.9)
        else:
            cv2.ellipse(a, c, (r, r if v["mass_shape"] == "round" else r // 2 + 3), 0, 0, 360, 0.9, -1)
        if v["mass_margin"] == "spiculated":
            for t in np.linspace(0, 2 * np.pi, 12):
                cv2.line(a, c, (int(c[0] + 2.2 * r * np.cos(t)), int(c[1] + 2.2 * r * np.sin(t))), 0.8, 1)
    if v.get("calc") == "present":
        for _ in range(12 if v["calc_distribution"] == "grouped" else 30):
            cv2.circle(a, (int(rng.integers(20, W - 20)), int(rng.integers(20, H - 20))), 1, 1.0, -1)
    return np.clip(a, 0, 1)

def main(out="smoke", n=160, n_ext=40, seed=0):
    rng = np.random.default_rng(seed); r0 = random.Random(seed)
    os.makedirs(f"{out}/img", exist_ok=True); os.makedirs(f"{out}/manifests", exist_ok=True)
    for fname, N, split_fn, ds in (("synthetic", n, lambda i: "val" if i % 5 == 0 else "train", "cbis_ddsm"),
                                   ("synthetic_ext", n_ext, lambda i: "test", "cdd_cesm")):
        recs = []
        for i in range(N):
            v = {"density": r0.choice("ABCD"), "mass": r0.choice(["absent", "present"]), "calc": r0.choice(["absent", "present"]),
                 "asymmetry": "absent", "distortion": "absent"}
            if v["mass"] == "present":
                v["mass_shape"] = r0.choice(["oval", "round", "irregular"]); v["mass_margin"] = r0.choice(["circumscribed", "spiculated"])
            if v["calc"] == "present":
                v["calc_morphology"] = r0.choice(["typically_benign", "fine_pleomorphic"]); v["calc_distribution"] = r0.choice(["grouped", "diffuse"])
            susp = v.get("mass_shape") == "irregular" or v.get("mass_margin") == "spiculated" or v.get("calc_morphology") == "fine_pleomorphic"
            v["birads"] = "4" if susp else ("2" if (v["mass"] == "present" or v["calc"] == "present") else "1")
            v["malignant"] = "malignant" if susp and r0.random() < .7 else "benign"
            views = {}
            for vp in ("CC", "MLO"):
                p = f"{out}/img/{fname}_{i}_{vp}.png"; cv2.imwrite(p, (draw(v, rng) * 255).astype(np.uint8)); views[vp] = p
            lab = C.encode(v)
            from cbmammo.reports import render
            recs.append({"sample_id": f"{fname}:{i}", "dataset": ds, "patient_id": f"{fname}{i // 2}", "study_id": str(i),
                         "side": "L", "split": split_fn(i), "views": views, "labels": lab, "values": C.decode(lab),
                         "report": render(C.decode(lab), "L", 3) if fname == "synthetic_ext" else None, "population": "synthetic"})
        with open(f"{out}/manifests/{fname}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    print("synthetic data written to", out)

if __name__ == "__main__":
    main()
