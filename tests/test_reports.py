import itertools, random
from cbmammo import concepts as C
from cbmammo.reports import render, extract, STYLES

def random_values(rng):
    v = {"density": rng.choice("ABCD"), "mass": rng.choice(["absent", "present"]), "calc": rng.choice(["absent", "present"]),
         "asymmetry": rng.choice(["absent", "present"]), "distortion": rng.choice(["absent", "present"]),
         "birads": rng.choice("12345")}
    if v["mass"] == "present":
        v["mass_shape"] = rng.choice(C.HEAD_BY_NAME["mass_shape"].classes); v["mass_margin"] = rng.choice(C.HEAD_BY_NAME["mass_margin"].classes)
    if v["calc"] == "present":
        v["calc_morphology"] = rng.choice(C.HEAD_BY_NAME["calc_morphology"].classes); v["calc_distribution"] = rng.choice(C.HEAD_BY_NAME["calc_distribution"].classes)
    return C.decode(C.encode(v))

def test_roundtrip():
    rng = random.Random(0); bad = []
    for _ in range(3000):
        v = random_values(rng); s = rng.randrange(STYLES); e = extract(render(v, "L", s))
        for k, val in v.items():
            if k != "malignant" and e.get(k) != val:
                bad.append((k, val, e.get(k), render(v, "L", s)))
    assert not bad, bad[:5]

def test_encode_hierarchy():
    lab = C.encode({"mass": "absent", "mass_margin": "spiculated"})
    assert lab["mass_margin"] == C.IGNORE
    lab = C.encode({"mass_margin": "spiculated"})
    assert lab["mass"] == 1
