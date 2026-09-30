"""BI-RADS (5th ed.) concept schema shared by every dataset.

Each breast-level sample carries one label per head. A label of -1 means
"unknown / not annotated" and is masked out of the loss. Descriptor heads are
children of a presence head: they are only supervised (and only meaningful)
when the parent finding is present.
"""
from __future__ import annotations

from dataclasses import dataclass

IGNORE = -1


@dataclass(frozen=True)
class Head:
    name: str
    classes: tuple
    parent: str | None = None          # presence head that gates this descriptor
    kind: str = "concept"              # "concept" | "target"

    @property
    def n(self) -> int:
        return len(self.classes)

    def idx(self, value) -> int:
        if value is None:
            return IGNORE
        try:
            return self.classes.index(value)
        except ValueError:
            return IGNORE


HEADS = (
    Head("density", ("A", "B", "C", "D")),
    Head("mass", ("absent", "present")),
    Head("mass_shape", ("oval", "round", "irregular"), parent="mass"),
    Head("mass_margin", ("circumscribed", "obscured", "microlobulated", "indistinct", "spiculated"), parent="mass"),
    Head("calc", ("absent", "present")),
    Head("calc_morphology", ("typically_benign", "amorphous", "coarse_heterogeneous",
                             "fine_pleomorphic", "fine_linear_branching"), parent="calc"),
    Head("calc_distribution", ("diffuse", "regional", "grouped", "linear", "segmental"), parent="calc"),
    Head("asymmetry", ("absent", "present")),
    Head("distortion", ("absent", "present")),
    # Targets: in the concept-bottleneck arm these are predicted FROM the concepts only.
    Head("birads", ("1", "2", "3", "4", "5"), kind="target"),
    Head("malignant", ("benign", "malignant"), kind="target"),
)
HEAD_BY_NAME = {h.name: h for h in HEADS}
CONCEPT_HEADS = tuple(h for h in HEADS if h.kind == "concept")
TARGET_HEADS = tuple(h for h in HEADS if h.kind == "target")
PRESENCE_HEADS = ("mass", "calc", "asymmetry", "distortion")


def empty_labels() -> dict:
    return {h.name: IGNORE for h in HEADS}


def encode(values: dict) -> dict:
    """String values -> integer labels, enforcing parent/child consistency."""
    lab = empty_labels()
    for k, v in values.items():
        if k in HEAD_BY_NAME:
            lab[k] = HEAD_BY_NAME[k].idx(v)
    for h in HEADS:
        if not h.parent:
            continue
        if lab[h.name] != IGNORE and lab[h.parent] == IGNORE:
            lab[h.parent] = 1                    # a descriptor implies presence
        if lab[h.parent] == 0:                   # parent absent -> descriptor undefined
            lab[h.name] = IGNORE
    return lab


def decode(labels: dict) -> dict:
    return {k: (HEAD_BY_NAME[k].classes[v] if v != IGNORE else None) for k, v in labels.items()}


# --------------------------------------------------------------------------
# Vocabulary normalisation: raw dataset strings -> schema classes.
# DDSM predates BI-RADS 5th ed.: LOBULATED was folded into OVAL, ILL_DEFINED is
# INDISTINCT, PLEOMORPHIC is FINE_PLEOMORPHIC, CLUSTERED is GROUPED.
# --------------------------------------------------------------------------
SHAPE_MAP = {"OVAL": "oval", "LOBULATED": "oval", "ROUND": "round", "IRREGULAR": "irregular"}
MARGIN_MAP = {"CIRCUMSCRIBED": "circumscribed", "OBSCURED": "obscured",
              "MICROLOBULATED": "microlobulated", "ILL_DEFINED": "indistinct",
              "INDISTINCT": "indistinct", "SPICULATED": "spiculated", "SPECULATED": "spiculated"}
CALC_MORPH_MAP = {
    "AMORPHOUS": "amorphous", "COARSE_HETEROGENEOUS": "coarse_heterogeneous",
    "HETEROGENEOUS": "coarse_heterogeneous",
    "PLEOMORPHIC": "fine_pleomorphic", "FINE_PLEOMORPHIC": "fine_pleomorphic",
    "FINE_LINEAR_BRANCHING": "fine_linear_branching", "LINEAR_BRANCHING": "fine_linear_branching",
    **{k: "typically_benign" for k in (
        "PUNCTATE", "ROUND_AND_REGULAR", "ROUND", "LUCENT_CENTER", "LUCENT_CENTERED", "COARSE",
        "DYSTROPHIC", "EGGSHELL", "RIM", "VASCULAR", "LARGE_RODLIKE", "MILK_OF_CALCIUM", "SKIN",
        "POPCORN", "SUTURE")},
}
CALC_DIST_MAP = {"DIFFUSE": "diffuse", "DIFFUSELY_SCATTERED": "diffuse", "SCATTERED": "diffuse",
                 "REGIONAL": "regional", "CLUSTERED": "grouped", "GROUPED": "grouped",
                 "LINEAR": "linear", "SEGMENTAL": "segmental"}
DENSITY_MAP = {"1": "A", "2": "B", "3": "C", "4": "D", "A": "A", "B": "B", "C": "C", "D": "D"}

# Severity order used to pick the most suspicious value when a breast has several lesions.
SEVERITY = {
    "mass_shape": ["oval", "round", "irregular"],
    "mass_margin": ["circumscribed", "obscured", "microlobulated", "indistinct", "spiculated"],
    "calc_morphology": ["typically_benign", "amorphous", "coarse_heterogeneous",
                        "fine_pleomorphic", "fine_linear_branching"],
    "calc_distribution": ["diffuse", "regional", "grouped", "linear", "segmental"],
    "birads": ["1", "2", "3", "4", "5"],
    "malignant": ["benign", "malignant"],
}


def is_missing(raw) -> bool:
    return raw is None or (isinstance(raw, float) and raw != raw) or str(raw).strip().upper() in {"", "NAN", "N/A", "NONE"}


def norm_token(raw) -> list:
    """'IRREGULAR-ARCHITECTURAL_DISTORTION' -> ['IRREGULAR', 'ARCHITECTURAL_DISTORTION']."""
    if is_missing(raw):
        return []
    s = str(raw).strip().upper().replace(" ", "_")
    return [t for t in s.replace(",", "-").split("-") if t]


def map_first(raw, table: dict):
    """First token (split on '-') that matches a key exactly, else by substring
    ('PARTIALLY_OBSCURED_LOBULATED' -> OBSCURED; longest key wins)."""
    toks = norm_token(raw)
    for t in toks:
        if t in table:
            return table[t]
    for t in toks:
        hits = [k for k in table if k in t]
        if hits:
            return table[max(hits, key=len)]
    return None


def most_severe(head: str, values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    order = SEVERITY.get(head)
    return max(vals, key=order.index) if order else vals[0]


def birads_group(raw):
    """BI-RADS 0-6 -> '1'..'5' (0 = incomplete -> unknown; 4A/4B/4C -> 4; 6 known cancer -> 5).
    Multi-lesion strings like '4$2' (CDD-CESM) -> the most severe category."""
    if is_missing(raw):
        return None
    if "$" in str(raw):
        return most_severe("birads", [birads_group(x) for x in str(raw).split("$")])
    s = str(raw).upper().replace("BI-RADS", "").replace("BIRADS", "").strip()
    if not s:
        return None
    d = s[0]
    if d in "12345":
        return d
    if d == "6":
        return "5"
    return None


def density_letter(raw):
    if is_missing(raw):
        return None
    s = str(raw).upper().replace("DENSITY", "").replace("ACR", "").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return DENSITY_MAP.get(s[:1]) if s else None
