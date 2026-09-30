"""Templated report rendering (training targets) and a rule-based slot extractor
(the measurement instrument for generated AND real reports).

The extractor must be validated against structured fields before any metric
built on it is reported (see evaluate.validate_extractor).
"""
from __future__ import annotations

import random
import re

from .concepts import HEAD_BY_NAME

DENSITY_TXT = {
    "A": "The breasts are almost entirely fatty (ACR A).",
    "B": "There are scattered areas of fibroglandular density (ACR B).",
    "C": "The breasts are heterogeneously dense, which may obscure small lesions (ACR C).",
    "D": "The breasts are extremely dense, which lowers the sensitivity of mammography (ACR D).",
}
MARGIN_TXT = {"circumscribed": "circumscribed", "obscured": "obscured", "microlobulated": "microlobulated",
              "indistinct": "indistinct", "spiculated": "spiculated"}
MORPH_TXT = {"typically_benign": "typically benign", "amorphous": "amorphous",
             "coarse_heterogeneous": "coarse heterogeneous", "fine_pleomorphic": "fine pleomorphic",
             "fine_linear_branching": "fine linear branching"}
BIRADS_TXT = {"1": "BI-RADS 1: negative.", "2": "BI-RADS 2: benign.", "3": "BI-RADS 3: probably benign.",
              "4": "BI-RADS 4: suspicious abnormality.", "5": "BI-RADS 5: highly suggestive of malignancy."}

STYLES = 4


def render(v: dict, side: str | None = None, style: int = 0) -> str:
    """Render a report from decoded concept values (None = not stated).

    `style` in 0..STYLES-1 varies surface form only (sentence order, phrasing),
    never content, so templated training cannot be solved by memorising one string.
    """
    side_txt = {"L": "left", "R": "right"}.get(side or "", "")
    sb = f"{side_txt} breast" if side_txt else "breast"
    parts: list[str] = []
    if v.get("density"):
        parts.append(DENSITY_TXT[v["density"]])
    findings: list[str] = []
    if v.get("mass") == "present":
        shape, margin = v.get("mass_shape"), v.get("mass_margin")
        desc = " ".join(x for x in [shape, "mass"] if x)
        if margin:
            desc += f" with {MARGIN_TXT[margin]} margins" if style % 2 == 0 else f", margins {MARGIN_TXT[margin]}"
        findings.append(f"There is an {desc} in the {sb}." if desc[0] in "aeiou" else f"There is a {desc} in the {sb}.")
    elif v.get("mass") == "absent":
        findings.append("No suspicious mass is seen.")
    if v.get("calc") == "present":
        m, d = v.get("calc_morphology"), v.get("calc_distribution")
        desc = " ".join(x for x in [MORPH_TXT.get(m), "calcifications"] if x)
        if d:
            desc += f" in a {d} distribution" if style < 2 else f", {d} distribution"
        findings.append(f"{desc[0].upper()}{desc[1:]} are present.")
    elif v.get("calc") == "absent":
        findings.append("No calcifications are seen.")
    if v.get("asymmetry") == "present":
        findings.append("An asymmetry is noted.")
    elif v.get("asymmetry") == "absent":
        findings.append("No asymmetry.")
    if v.get("distortion") == "present":
        findings.append("Architectural distortion is present.")
    elif v.get("distortion") == "absent":
        findings.append("No architectural distortion.")
    if style in (1, 3):
        findings = list(reversed(findings))
    parts += findings
    if v.get("birads"):
        parts.append(("Impression: " if style < 2 else "Assessment: ") + BIRADS_TXT[v["birads"]])
    return " ".join(parts)


def render_random(v: dict, side=None, rng: random.Random | None = None) -> str:
    rng = rng or random
    return render(v, side, rng.randrange(STYLES))


# --------------------------------------------------------------------------
# Slot extractor. Returns None for a slot when the text does not mention it;
# the coverage guard in evaluate.py reports how often that happens.
# --------------------------------------------------------------------------
_NEG = r"\b(?:no|without|negative for|absence of)\b[^.;:\n]*?"
_P = {
    "density": [(r"almost entirely fatty|fatty breast|\bACR\s*A\b|density\s*a\b|\btype\s*a\b", "A"),
                (r"scattered (?:areas of )?fibroglandular|\bACR\s*B\b|density\s*b\b|\btype\s*b\b", "B"),
                (r"heterogeneously dense|\bACR\s*C\b|density\s*c\b|\btype\s*c\b", "C"),
                (r"extremely dense|\bACR\s*D\b|density\s*d\b|\btype\s*d\b", "D")],
    "mass_shape": [(r"\birregular\b", "irregular"), (r"\bround\b(?!\w)", "round"), (r"\boval\b|\blobulated\b", "oval")],
    "mass_margin": [(r"spiculat|speculat", "spiculated"), (r"microlobulat", "microlobulated"),
                    (r"indistinct|ill[- ]defined", "indistinct"), (r"obscured", "obscured"),
                    (r"circumscribed|well[- ]defined", "circumscribed")],
    "calc_morphology": [(r"fine[- ]linear|linear[- ]branching", "fine_linear_branching"),
                        (r"pleomorphic", "fine_pleomorphic"),
                        (r"coarse heterogeneous", "coarse_heterogeneous"), (r"amorphous", "amorphous"),
                        (r"typically benign|punctate|pop ?corn|vascular calc|dystrophic|\brim\b|eggshell|milk of calcium|rod[- ]like|macro ?calc|calcific foci|benign calc",
                         "typically_benign")],
    "calc_distribution": [(r"segmental", "segmental"), (r"linear distribution|linearly distributed|, linear distribution", "linear"),
                          (r"grouped|cluster", "grouped"), (r"regional", "regional"), (r"diffuse|scattered", "diffuse")],
}
_BIRADS = re.compile(r"bi-?rads\s*(?:category\s*)?[:\-]?\s*([0-6])", re.I)


def _first(patterns, text):
    for pat, val in patterns:
        if re.search(pat, text, re.I):
            return val
    return None


def _presence(text, pos, neg_word, qualified_neg_is_unknown=False):
    """Sentence-level presence. A positive mention in any non-negated sentence wins
    ('No suspicious microcalcifications. Benign macrocalcifications.' -> present).
    With qualified_neg_is_unknown, 'no SUSPICIOUS x' says nothing about benign x -> None."""
    sents = re.split(r"(?<=[.;:\n])\s+", text)
    neg = [s for s in sents if re.search(_NEG + neg_word, s, re.I)]
    if any(re.search(pos, s, re.I) for s in sents if s not in neg and not re.search(r"obscure small|dense", s, re.I)):
        return "present"
    if neg:
        if qualified_neg_is_unknown and all(re.search(r"suspicious|malignant[- ]appearing", s, re.I) for s in neg):
            return None
        return "absent"
    return None


def extract(text: str) -> dict:
    t = " ".join(text.split())
    out = {k: None for k in HEAD_BY_NAME}
    out["density"] = _first(_P["density"], t)
    out["mass"] = _presence(t, r"\bmass(?:es)?\b", r"mass")
    out["calc"] = _presence(t, r"calcific", r"(?:micro|macro)?calcific", qualified_neg_is_unknown=True)
    out["asymmetry"] = _presence(t, r"asymmetr", r"asymmetr")
    out["distortion"] = _presence(t, r"architectural distortion|\bdistortion\b", r"(?:architectural )?distortion")
    # Descriptors are only read from the sentence that mentions the finding, to avoid
    # e.g. "linear" in a calcification sentence being read as a mass margin.
    sents = re.split(r"(?<=[.;])\s+", t)
    mass_s = " ".join(s for s in sents if re.search(r"\bmass", s, re.I) and not re.search(_NEG + "mass", s, re.I))
    calc_s = " ".join(s for s in sents if re.search(r"calcific", s, re.I) and not re.search(_NEG + r"(?:micro|macro)?calcific", s, re.I))
    if mass_s:
        out["mass_shape"] = _first(_P["mass_shape"], mass_s)
        out["mass_margin"] = _first(_P["mass_margin"], mass_s)
    if calc_s:
        out["calc_morphology"] = _first(_P["calc_morphology"], calc_s)
        out["calc_distribution"] = _first(_P["calc_distribution"], calc_s)
    m = _BIRADS.findall(t)
    if m:
        d = max(m)                      # most severe category mentioned
        out["birads"] = {"0": None, "6": "5"}.get(d, d)
    out["malignant"] = None             # not a report slot
    return out
