"""Change/new-candidate concept schema for paper 2 (temporal descriptor change).

Kept separate from concepts.py (paper 1's static BI-RADS schema) because these heads
describe a DIFFERENT thing -- what changed between two exams for a given presence head --
not a value of the current exam, and because the class taxonomy was deliberately
simplified after seeing the final embed_temporal.jsonl counts (see
paper2_brief_temporal_change.md section 11): 'increased' has only n=39 in the whole
2026-09 manifest, too thin to train or evaluate on its own, so the trained/evaluated
change head is 3-way (stable/resolved/changed), not the manifest's raw 4-way field.
"""
from __future__ import annotations

IGNORE = -1

# One change head and one new-candidate head per PRESENCE head (mirrors concepts.PRESENCE_HEADS).
PRESENCE_HEADS = ("mass", "calc", "asymmetry", "distortion")
# Trained/evaluated temporal heads = the four per-descriptor heads + one breast-level POOLED head.
# The pooled label exists because per-descriptor labels are far too sparse for mass/calc (train:
# 3 and 2 'resolved', 24 and 32 'changed'); it asks "did ANY annotated finding on this breast
# change?" and pools every labelled descriptor. Precedence: changed > resolved > stable.
TEMPORAL_HEADS = PRESENCE_HEADS + ("pooled",)

CHANGE_CLASSES = ("stable", "resolved", "changed")   # 'changed' = raw increased+decreased merged
# raw embed_temporal.jsonl 4-way value -> trained 3-way class
_CHANGE_MERGE = {"stable": "stable", "resolved": "resolved", "increased": "changed", "decreased": "changed"}

NEW_CLASSES = ("not_new", "new")


def encode_change(raw_change: dict) -> dict:
    """embed_temporal.jsonl 'change' dict (per PRESENCE head: raw 4-way string or None)
    -> {head: int label}, 3-way (CHANGE_CLASSES), IGNORE where the manifest has no code."""
    out = {}
    for h in PRESENCE_HEADS:
        v = raw_change.get(h)
        out[h] = CHANGE_CLASSES.index(_CHANGE_MERGE[v]) if v else IGNORE
    vals = {out[h] for h in PRESENCE_HEADS if out[h] != IGNORE}
    ch, rs, st = (CHANGE_CLASSES.index(k) for k in ("changed", "resolved", "stable"))
    out["pooled"] = ch if ch in vals else rs if rs in vals else st if st in vals else IGNORE
    return out


def decode_change(labels: dict) -> dict:
    return {h: (CHANGE_CLASSES[v] if v != IGNORE else None) for h, v in labels.items()}


def encode_new(raw_change: dict, raw_new_candidate: dict, include_weak: bool = False) -> tuple:
    """embed_temporal.jsonl 'change' + 'new_candidate' dicts -> ({head: int label}, {head: float weight}).

    Negatives and positives come from two DIFFERENT fields, which is the whole point of this
    function (get this wrong and the head trains on positives only):
      - a head with an explicit `changed` code (stable/resolved/increased/decreased) means the
        radiologist successfully compared it to a prior finding -> confirmed NOT new (label 0,
        weight 1.0). This is the only source of clean negative supervision for this head.
      - new_candidate 'strong' (no finding at all on that side in the prior exam) -> label 1,
        weight 1.0.
      - new_candidate 'weak' (prior exam has same-side findings but none coded -- ambiguous, see
        paper2_brief_temporal_change.md section 4/9) -> label 1, weight 0.0 by default (i.e.
        effectively excluded from the loss) unless include_weak=True (weight 0.3 -- a judgment
        call to sweep, not a validated choice).
      - neither field populated for a head -> IGNORE, weight 0.0 (no evidence either way).
    """
    labels, weights = {}, {}
    for h in PRESENCE_HEADS:
        if raw_change.get(h):
            labels[h], weights[h] = NEW_CLASSES.index("not_new"), 1.0
            continue
        tag = raw_new_candidate.get(h)
        if tag == "strong":
            labels[h], weights[h] = NEW_CLASSES.index("new"), 1.0
        elif tag == "weak":
            labels[h] = NEW_CLASSES.index("new")
            weights[h] = 0.3 if include_weak else 0.0
        else:
            labels[h], weights[h] = IGNORE, 0.0
    # pooled: 'new' if any descriptor is a (weighted) new candidate, else 'not_new' if any
    # descriptor carries an explicit change code, else unlabelled.
    new_i, not_i = NEW_CLASSES.index("new"), NEW_CLASSES.index("not_new")
    new_w = [weights[h] for h in PRESENCE_HEADS if labels[h] == new_i and weights[h] > 0]
    if new_w:
        labels["pooled"], weights["pooled"] = new_i, max(new_w)
    elif any(labels[h] == not_i for h in PRESENCE_HEADS):
        labels["pooled"], weights["pooled"] = not_i, 1.0
    else:
        labels["pooled"], weights["pooled"] = IGNORE, 0.0
    return labels, weights
