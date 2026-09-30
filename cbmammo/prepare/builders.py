"""Build one breast-level JSONL manifest per dataset.

Record schema (one line per breast = one exam side):
  sample_id, dataset, patient_id, study_id, side ('L'|'R'), split ('train'|'val'|'test'),
  views: {"CC": path|None, "MLO": path|None},
  values: {head: class-string|None},  labels: {head: int (-1 = unknown)},
  report: real report text or None, population: str

Column names follow each dataset's public release. Every builder prints what it
dropped and why; check those counts on the first run on the server. Anything
marked VERIFY depends on a detail of the local copy (folder layout, code tables)
that should be confirmed once.
"""
from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd

from .. import concepts as C
from ..reports import extract

POPULATION = {"cbis_ddsm": "USA", "embed": "USA", "embed_temporal": "USA", "vindr": "Vietnam", "cmmd": "China",
              "cdd_cesm": "Egypt", "inbreast": "Portugal", "mmc_astana": "Kazakhstan"}


def hsplit(patient_id: str, val_frac=0.1, test_frac=0.0, salt="cbm") -> str:
    """Deterministic patient-level split by hash (used where no official split exists)."""
    h = int(hashlib.sha1(f"{salt}:{patient_id}".encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "val"
    return "train"


def finalize(rec: dict) -> dict:
    rec["labels"] = C.encode(rec["values"])
    rec["values"] = C.decode(rec["labels"])          # canonical, consistency-enforced
    rec["population"] = POPULATION[rec["dataset"]]
    return rec


def write(records: list, out: str) -> None:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    sp = Counter(r["split"] for r in records)
    lab = {h.name: sum(r["labels"][h.name] != C.IGNORE for r in records) for h in C.HEADS}
    print(f"[{os.path.basename(out)}] {len(records)} breasts  splits={dict(sp)}")
    print("  labelled per head:", lab)


def _merge_lesions(lesions: list) -> dict:
    """Breast-level values from lesion-level rows: presence = any; descriptors = most severe."""
    v = {h.name: None for h in C.HEADS}
    for pres in C.PRESENCE_HEADS:
        vals = [l.get(pres) for l in lesions if l.get(pres) is not None]
        v[pres] = "present" if "present" in vals else ("absent" if vals else None)
    for h in C.HEADS:
        if h.name in C.PRESENCE_HEADS:
            continue
        v[h.name] = C.most_severe(h.name, [l.get(h.name) for l in lesions])
    return v


# ------------------------------------------------------------------ CBIS-DDSM
def cbis_ddsm(root: str, out: str) -> list:
    """root = folder with the 4 *_case_description_*.csv files and the TCIA image tree.

    CBIS-DDSM contains only breasts WITH an abnormality, so 'mass' / 'calc' absent
    labels exist only across the two case types (a mass case with no calc row is
    NOT evidence of no calcification) -> leave the other presence head unknown.
    VERIFY: the TCIA tree is <root>/**/<SeriesFolder>/.../*.dcm where SeriesFolder is
    the first component of 'image file path' (e.g. Mass-Training_P_00001_LEFT_CC).
    """
    csvs = sorted(glob.glob(os.path.join(root, "**", "*case_description*.csv"), recursive=True))
    assert csvs, f"no CBIS-DDSM case_description CSVs under {root}"
    dcm_index = defaultdict(list)
    for p in glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True) + glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True):
        parts = p.split(os.sep)
        for part in parts:
            if re.match(r"(Mass|Calc)-(Training|Test)_P_\d+_(LEFT|RIGHT)_(CC|MLO)$", part):
                dcm_index[part].append(p)
                break
    breasts = defaultdict(lambda: {"lesions": [], "views": {"CC": None, "MLO": None}})
    for csv in csvs:
        df = pd.read_csv(csv)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        is_mass = "mass" in os.path.basename(csv).lower()
        split = "test" if "test" in os.path.basename(csv).lower() else "train"
        for _, r in df.iterrows():
            side = "L" if str(r["left_or_right_breast"]).upper().startswith("L") else "R"
            key = (r["patient_id"], side, split)
            b = breasts[key]
            les = {"birads": C.birads_group(r.get("assessment")),
                   "malignant": "malignant" if str(r.get("pathology", "")).upper() == "MALIGNANT" else "benign",
                   "density": C.density_letter(r.get("breast_density", r.get("breast_density_"))),
                   "asymmetry": None, "distortion": None}
            if is_mass:
                toks = C.norm_token(r.get("mass_shape"))
                les["mass"] = "present"
                les["mass_shape"] = C.map_first(r.get("mass_shape"), C.SHAPE_MAP)
                les["mass_margin"] = C.map_first(r.get("mass_margins"), C.MARGIN_MAP)
                if any("ASYMMETR" in t for t in toks):
                    les["asymmetry"] = "present"
                if "ARCHITECTURAL_DISTORTION" in toks:
                    les["distortion"] = "present"
            else:
                les["calc"] = "present"
                les["calc_morphology"] = C.map_first(r.get("calc_type"), C.CALC_MORPH_MAP)
                les["calc_distribution"] = C.map_first(r.get("calc_distribution"), C.CALC_DIST_MAP)
            b["lesions"].append(les)
            series = str(r["image_file_path"]).split("/")[0]
            files = dcm_index.get(series, [])
            if files:
                b["views"][str(r["image_view"]).upper()] = sorted(files)[0]
    recs, miss = [], 0
    for (pid, side, split), b in breasts.items():
        if not any(b["views"].values()):
            miss += 1
            continue
        v = _merge_lesions(b["lesions"])
        v["density"] = next((l["density"] for l in b["lesions"] if l.get("density")), None)
        recs.append(finalize({
            "sample_id": f"cbis:{pid}:{side}:{split}", "dataset": "cbis_ddsm", "patient_id": pid,
            "study_id": pid, "side": side,
            "split": split if split == "test" else hsplit(pid), "views": b["views"],
            "values": v, "report": None}))
    print(f"cbis_ddsm: dropped {miss} breasts with no resolvable DICOM")
    write(recs, out)
    return recs


# ------------------------------------------------------------------ VinDr-Mammo
_VINDR_CAT = {"Mass": ("mass", "present"), "Suspicious Calcification": ("calc", "present"),
              "Asymmetry": ("asymmetry", "present"), "Focal Asymmetry": ("asymmetry", "present"),
              "Global Asymmetry": ("asymmetry", "present"),
              "Architectural Distortion": ("distortion", "present")}


def vindr(root: str, out: str) -> list:
    """root = physionet vindr-mammo/1.0.0 (finding_annotations.csv, breast-level_annotations.csv, images/).

    VinDr provides finding categories, BI-RADS and density; no shape/margin/morphology,
    no pathology. A breast with 'No Finding' gives absent labels for all four presence
    heads (every finding category was annotated exhaustively by the readers).
    """
    bl = pd.read_csv(os.path.join(root, "breast-level_annotations.csv"))
    fa = pd.read_csv(os.path.join(root, "finding_annotations.csv"))
    findings = defaultdict(set)
    for _, r in fa.iterrows():
        try:
            cats = ast.literal_eval(r["finding_categories"])
        except Exception:
            cats = [str(r["finding_categories"])]
        for c in cats:
            if c in _VINDR_CAT:
                findings[(r["study_id"], r["laterality"])].add(_VINDR_CAT[c])
    breasts = defaultdict(lambda: {"views": {"CC": None, "MLO": None}})
    for _, r in bl.iterrows():
        k = (r["study_id"], r["laterality"])
        b = breasts[k]
        b.update(birads=C.birads_group(r["breast_birads"]), density=C.density_letter(r["breast_density"]),
                 split="test" if str(r["split"]).lower().startswith("test") else "train")
        vp = str(r["view_position"]).upper()
        if vp in ("CC", "MLO"):
            cands = [os.path.join(root, d, r["study_id"], f"{r['image_id']}{ext}")
                     for d in ("images", "images2") for ext in (".dicom", ".dcm", ".jpg", ".png")]
            b["views"][vp] = next((c for c in cands if os.path.exists(c)), cands[0])
    recs = []
    for (sid, side), b in breasts.items():
        v = {p: "absent" for p in C.PRESENCE_HEADS}
        for head, val in findings.get((sid, side), ()):
            v[head] = val
        v.update(density=b["density"], birads=b["birads"])
        recs.append(finalize({
            "sample_id": f"vindr:{sid}:{side}", "dataset": "vindr", "patient_id": sid, "study_id": sid,
            "side": side, "split": b["split"] if b["split"] == "test" else hsplit(sid),
            "views": b["views"], "values": v, "report": None}))
    write(recs, out)
    return recs


# ------------------------------------------------------------------ CMMD
def _dicom_view(path):
    import pydicom
    d = pydicom.dcmread(path, stop_before_pixels=True)
    lat = str(getattr(d, "ImageLaterality", "") or getattr(d, "Laterality", "")).upper()[:1]
    vp = str(getattr(d, "ViewPosition", "")).upper()
    if not vp and hasattr(d, "ViewCodeSequence"):
        meaning = str(d.ViewCodeSequence[0].CodeMeaning).lower()
        vp = "CC" if "cranio" in meaning else ("MLO" if "oblique" in meaning else "")
    return lat, vp


def cmmd(root: str, out: str, clinical_xlsx: str | None = None) -> list:
    """root = TCIA CMMD tree (<root>/CMMD/<ID1>/**/*.dcm) + CMMD_clinicaldata_revision.xlsx.

    Labels: abnormality (mass / calcification / both) and biopsy pathology. CMMD has
    no normal breasts, no descriptors, no BI-RADS -> it is an EXTERNAL TEST set for
    presence + malignancy only (population: China).
    """
    xl = clinical_xlsx or next(iter(glob.glob(os.path.join(root, "**", "CMMD_clinicaldata*.xlsx"), recursive=True)), None)
    assert xl, "CMMD clinical xlsx not found"
    df = pd.read_excel(xl)
    prepared = os.path.join(root, "cmmd_dataset.csv")
    vmap = defaultdict(dict)
    if os.path.exists(prepared):
        pv = pd.read_csv(prepared)
        for _, q in pv.iterrows():
            m = re.search(r"(D\d-\d+)", str(q["file_path"]))
            fp = os.path.join(os.path.dirname(root.rstrip("/")), str(q["file_path"]).lstrip("/"))
            if m and os.path.exists(fp):
                vmap[(m.group(1), str(q["laterality"]).upper()[:1])][str(q["view"]).upper()] = fp
    recs = []
    for _, r in df.iterrows():
        pid, side = str(r["ID1"]), str(r["LeftRight"]).upper()[:1]
        views = {"CC": None, "MLO": None, **vmap.get((pid, side), {})}
        for p in ([] if vmap else glob.glob(os.path.join(root, "**", pid, "**", "*.dcm"), recursive=True)):
            try:
                lat, vp = _dicom_view(p)
            except Exception:
                continue
            if lat == side and vp in views and views[vp] is None:
                views[vp] = p
        ab = str(r.get("abnormality", "")).lower()
        v = {"mass": "present" if "mass" in ab or "both" in ab else None,
             "calc": "present" if "calc" in ab or "both" in ab else None,
             "malignant": "malignant" if str(r.get("classification", "")).lower().startswith("malig") else "benign"}
        recs.append(finalize({
            "sample_id": f"cmmd:{pid}:{side}", "dataset": "cmmd", "patient_id": pid, "study_id": pid,
            "side": side, "split": "test", "views": views, "values": v, "report": None}))
    write(recs, out)
    return recs


# ------------------------------------------------------------------ CDD-CESM
def _read_docx(path):
    import docx
    return "\n".join(p.text for p in docx.Document(path).paragraphs)


def _dm_section(text: str) -> str:
    """Keep the conventional-mammography part of a CDD-CESM report (drop the contrast part).
    VERIFY on a few reports: sections are headed by lines containing 'MAMMOGRAPHY' / 'CONTRAST'."""
    t = text
    m = re.search(r"(contrast[- ]enhanced|CESM|after (?:contrast|IV))", t, re.I)
    return t[:m.start()] if m and m.start() > 50 else t


def _side_section(text: str, side: str) -> str:
    """Take the paragraph(s) about one breast if the report is split by side."""
    key = "right" if side == "R" else "left"
    other = "left" if side == "R" else "right"
    m = re.search(rf"{key} breast[:\s].*?(?={other} breast[:\s]|$)", text, re.I | re.S)
    return m.group(0) if m else text


def cdd_cesm(root: str, out: str, annotations_xlsx: str | None = None, reports_dir: str | None = None,
             images_dir: str | None = None) -> list:
    """CDD-CESM (TCIA). LOW-ENERGY (DM-equivalent) images only; all breasts held out (split='test').

    Sheets of Radiology-manual-annotations.xlsx: 'all' (one row per image: density, BI-RADS,
    pathology), 'mass_description' (Mass shape / Mass margin columns), 'calcifications',
    'asymmetry', 'distortion' (descriptors in free-text 'Findings'). Presence of a finding =
    the image appears on that sheet; a 'Normal' image = all four absent.
    BI-RADS strings like '4$2' list several lesions -> most severe.
    """
    xl = annotations_xlsx or next(iter(glob.glob(os.path.join(root, "**", "*annotation*.xlsx"), recursive=True)), None)
    assert xl, "CDD-CESM annotation xlsx not found"
    x = pd.ExcelFile(xl)
    allv = pd.read_excel(x, "all")
    dm = allv[allv["Type"].astype(str).str.upper() == "DM"]
    sheets = {s: pd.read_excel(x, s) for s in ("mass_description", "calcifications", "asymmetry", "distortion") if s in x.sheet_names}
    imgs = {os.path.splitext(os.path.basename(p))[0].strip(): p
            for p in glob.glob(os.path.join(images_dir or root, "**", "*.jp*g"), recursive=True)}
    rep_files = {}
    for p in glob.glob(os.path.join(reports_dir or root, "**", "*.docx"), recursive=True):
        bn = os.path.basename(p)
        m = re.match(r"P(\d+)", bn)
        if m and not bn.startswith("~$"):
            rep_files[m.group(1)] = p
    per_img = defaultdict(list)                    # image name -> lesion dicts
    for sname, d in sheets.items():
        for _, r in d.iterrows():
            name = str(r["Image_name"]).strip()
            find = "" if C.is_missing(r.get("Findings")) else str(r.get("Findings"))
            l = {}
            if sname == "mass_description":
                l["mass"] = "present"
                l["mass_shape"] = C.map_first(r.get("Mass shape"), C.SHAPE_MAP)
                l["mass_margin"] = C.map_first(r.get("Mass margin"), C.MARGIN_MAP)
            elif sname == "calcifications":
                e = extract(find.replace("$", ". ")) if find else {}
                l["calc"] = "present"
                l["calc_morphology"], l["calc_distribution"] = e.get("calc_morphology"), e.get("calc_distribution")
            elif sname == "asymmetry":
                l["asymmetry"] = "present"
            elif sname == "distortion":
                l["distortion"] = "present"
            per_img[name].append(l)
    breasts = defaultdict(lambda: {"lesions": [], "views": {"CC": None, "MLO": None}, "normal": []})
    for _, r in dm.iterrows():
        name = str(r["Image_name"]).strip()
        pid = re.sub(r"\D", "", str(r["Patient_ID"]))
        side = str(r["Side"]).upper()[:1]
        vp = str(r["View"]).upper()
        b = breasts[(pid, side)]
        if vp in b["views"] and name in imgs:
            b["views"][vp] = imgs[name]
        pth = str(r["Pathology Classification/ Follow up"]).lower()
        base = {"birads": C.birads_group(r["BIRADS"]), "density": C.density_letter(r["Breast density (ACR)"]),
                "malignant": "malignant" if "malig" in pth else "benign"}
        b["normal"].append("normal" in pth)
        b["lesions"].append(base)
        b["lesions"] += per_img.get(name, [])
    recs = []
    for (pid, side), b in breasts.items():
        v = _merge_lesions(b["lesions"])
        v["density"] = next((l["density"] for l in b["lesions"] if l.get("density")), None)
        if all(b["normal"]):
            for p in C.PRESENCE_HEADS:
                v[p] = "absent"
        else:   # annotated abnormal breast: a finding not on its sheet is absent
            for p in C.PRESENCE_HEADS:
                v[p] = v[p] or "absent"
        report = None
        if pid in rep_files:
            try:
                report = _side_section(_dm_section(_read_docx(rep_files[pid])), side)
            except Exception as ex:
                print("report read failed", pid, ex)
        recs.append(finalize({
            "sample_id": f"cdd:{pid}:{side}", "dataset": "cdd_cesm", "patient_id": pid, "study_id": pid,
            "side": side, "split": "test", "views": b["views"], "values": v, "report": report}))
    print(f"cdd_cesm: {len(recs)} breasts, {sum(any(r['views'].values()) for r in recs)} with images, "
          f"{sum(r['report'] is not None for r in recs)} with a real report")
    write(recs, out)
    return recs


# ------------------------------------------------------------------ EMBED
_EMBED_ASSES = {"N": "1", "B": "2", "P": "3", "S": "4", "M": "5", "K": "5", "A": None, "X": None}

# EMBED open-data descriptor codes -> BI-RADS lexicon (from AWS_Open_Data_Clinical_Legend.csv; the legend text has typos such as
# 'Focal asymetry', so codes are mapped directly rather than by string matching). Lymph nodes (N, Y) and the asymmetric tubular
# structure (T) are not mapped to any concept; codes without a legend entry are ignored.
EMBED_SHAPE = {"O": "oval", "R": "round", "X": "irregular"}
EMBED_MASS_CODES = {"G", "O", "R", "X"}                       # generic, oval, round, irregular mass
EMBED_ASYM = {"B", "F", "S", "V"}                             # global, focal, (plain), developing asymmetry
EMBED_DISTORTION = {"A", "Q"}                                 # architectural distortion, questioned distortion
EMBED_MARGIN = {"D": "circumscribed", "U": "obscured", "M": "microlobulated", "I": "indistinct", "S": "spiculated"}
EMBED_CALC_MORPH = {"A": "amorphous", "B": "fine_linear_branching", "F": "fine_linear_branching", "H": "coarse_heterogeneous",
                    "I": "fine_pleomorphic", "K": "fine_pleomorphic", **{c1: "typically_benign" for c1 in "CDELMOPQRSVJ9"}}
EMBED_CALC_DIST = {"C": "grouped", "G": "grouped", "D": "diffuse", "R": "regional", "L": "linear", "S": "segmental"}


def _legend(legend_csv):
    """Map (field, code) -> meaning from clinical_legend.csv. VERIFY the column names once."""
    if not legend_csv or not os.path.exists(legend_csv):
        return {}
    lg = pd.read_csv(legend_csv)
    cols = [c.lower() for c in lg.columns]
    # EMBED open data legend: 'Header in export', 'Discription', 'Code', 'Meaning'
    f = lg.columns[next(i for i, c in enumerate(cols) if any(t in c for t in ("field", "column", "variable", "header")))]
    k = lg.columns[next(i for i, c in enumerate(cols) if c == "code" or "code" in c or "value" in c)]
    m = lg.columns[next(i for i, c in enumerate(cols) if "desc" in c or "meaning" in c or "label" in c)]
    return {(str(r[f]).strip().lower(), str(r[k]).strip()): str(r[m]) for _, r in lg.iterrows()}


_EMBED_VENDOR_HOLDOUT = ("ge", "fuji")   # vendors held out entirely from training as a perception-shift control


def _vendor_tag(manufacturer) -> str:
    s = str(manufacturer).upper()
    if "HOLOGIC" in s or "LORAD" in s:
        return "hologic"
    if s.startswith("GE ") or "GENERAL ELECTRIC" in s:
        return "ge"
    if "FUJIFILM" in s or "FUJI" in s:
        return "fuji"
    return "other"


def _embed_lesion(r) -> dict:
    """Per-finding-row descriptor dict, decoded from one EMBED_OpenData_clinical.csv row.
    Pure extraction from `embed()`'s former inline loop body (no behaviour change) so that
    `embed_temporal()` can decode a lesion the same way without duplicating the mapping logic."""
    codes = lambda col: [] if C.is_missing(r.get(col)) else [c1.strip() for c1 in str(r.get(col)).split(",") if c1.strip()]
    sh, mg, cf, cd = codes("massshape"), codes("massmargin"), codes("calcfind"), codes("calcdistri")
    worst = lambda head, vals: C.most_severe(head, [v for v in vals if v]) if any(vals) else None
    l = {"birads": _EMBED_ASSES.get(str(r.get("asses", "")).strip().upper()[:1]),
         "density": C.density_letter(r.get("tissueden")),
         "mass_shape": worst("mass_shape", [EMBED_SHAPE.get(c1) for c1 in sh]),
         "mass_margin": worst("mass_margin", [EMBED_MARGIN.get(c1) for c1 in mg]),
         "calc_morphology": worst("calc_morphology", [EMBED_CALC_MORPH.get(c1) for c1 in cf]),
         "calc_distribution": worst("calc_distribution", [EMBED_CALC_DIST.get(c1) for c1 in cd])}
    l["mass"] = "present" if (any(c1 in EMBED_MASS_CODES for c1 in sh) or l["mass_margin"]) else None
    l["calc"] = "present" if (cf or cd) else None
    l["asymmetry"] = "present" if any(c1 in EMBED_ASYM for c1 in sh) else None
    l["distortion"] = "present" if any(c1 in EMBED_DISTORTION for c1 in sh) else None
    if l["birads"] == "1":                                  # negative exam: nothing present
        for p in C.PRESENCE_HEADS:
            l[p] = "absent"
    sev = r.get("path_severity")
    if not C.is_missing(sev):
        l["malignant"] = "malignant" if int(float(sev)) in (0, 1) else "benign"
    return l


def embed(root: str, out: str, clinical_csv=None, metadata_csv=None, legend_csv=None,
          clinical_reduced_csv=None, path_prefix: str | None = None) -> list:
    """EMBED open data. Joins clinical (finding-level) to metadata (image-level) on
    (empi_anon, acc_anon, side). Descriptor codes are decoded via clinical_legend.csv.

    path_severity: 0 invasive, 1 non-invasive cancer -> malignant; 2-5 -> benign/negative
    (VERIFY against the EMBED README for your release). Only 2D FFDM CC/MLO images are kept.

    Vendor hold-out (perception-shift positive control, decided 2026-09-25): ANY patient with
    at least one non-Hologic (GE or Fujifilm) 2D exam is excluded from train/val/test entirely --
    ALL of their records (including their Hologic exams) go to split 'test_embed_ge' or
    'test_embed_fuji' instead. This costs training data (their Hologic exams too) but keeps the
    hold-out a clean, patient-disjoint test of the vendor/perception shift; it also lets a mixed-
    vendor patient's Hologic and non-Hologic exams (both now in the hold-out) be compared as a
    within-patient paired check (embed_meta['vendor'] on each record says which is which).
    A patient with both GE and Fujifilm exams is tagged 'ge' (GE has far more malignant cases).
    """
    cl = pd.read_csv(clinical_csv or glob.glob(os.path.join(root, "**", "*clinical*.csv"), recursive=True)[0], low_memory=False)
    md = pd.read_csv(metadata_csv or glob.glob(os.path.join(root, "**", "*metadata*.csv"), recursive=True)[0], low_memory=False)
    lg = _legend(legend_csv or next(iter(glob.glob(os.path.join(root, "**", "*legend*.csv"), recursive=True)), None))

    # ---- vendor per exam (2D, non-magnification rows only, matching the view filter below) and
    # per-patient hold-out tag, computed from metadata BEFORE any other filtering narrows `md`.
    mdv = md.copy()
    if "FinalImageType" in mdv.columns:
        mdv = mdv[mdv["FinalImageType"].astype(str).str.upper() == "2D"]
    if "spot_mag" in mdv.columns:
        mdv = mdv[mdv["spot_mag"].isna()]
    exam_vendor = mdv.groupby("acc_anon")["Manufacturer"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0]).map(_vendor_tag)
    pv = mdv[["empi_anon", "acc_anon"]].drop_duplicates()
    pv["vendor"] = pv["acc_anon"].map(exam_vendor)
    patient_vendors = pv.groupby("empi_anon")["vendor"].agg(set).to_dict()

    def holdout_tag(empi_anon):
        vs = patient_vendors.get(empi_anon, set())
        for v in _EMBED_VENDOR_HOLDOUT:            # 'ge' checked before 'fuji': see docstring
            if v in vs:
                return v
        return None

    # ---- descriptive metadata for later subgroup analysis (not used in the split/label logic)
    race_by_patient = {}
    rc_path = clinical_reduced_csv or next(iter(glob.glob(os.path.join(root, "**", "*clinical_reduced*.csv"), recursive=True)), None)
    if rc_path and os.path.exists(rc_path):
        rc = pd.read_csv(rc_path, usecols=lambda c: c in ("empi_anon", "RACE_DESC"), low_memory=False)
        race_by_patient = rc.dropna(subset=["RACE_DESC"]).drop_duplicates("empi_anon").set_index("empi_anon")["RACE_DESC"].to_dict()

    def dec(field, code, table, head=None):
        """Decode a (possibly multi-valued, comma-separated) code; several values -> the most severe."""
        if C.is_missing(code):
            return None
        vals = []
        for c1 in str(code).split(","):
            c1 = c1.strip()
            if c1:
                vals.append(C.map_first(lg.get((field, c1), c1), table))
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return C.most_severe(head, vals) if head and len(vals) > 1 else vals[0]

    if "FinalImageType" in md.columns:
        md = md[md["FinalImageType"].astype(str).str.upper() == "2D"]
    md = md[md["ViewPosition"].astype(str).str.upper().isin(["CC", "MLO"])]
    if "spot_mag" in md.columns:
        md = md[md["spot_mag"].isna()]
    lat_col = "ImageLateralityFinal" if "ImageLateralityFinal" in md.columns else "ImageLaterality"
    views = defaultdict(lambda: {"CC": None, "MLO": None})
    for _, r in md.iterrows():
        k = (r["empi_anon"], r["acc_anon"], str(r[lat_col]).upper()[:1])
        p = str(r["anon_dicom_path"])
        if path_prefix:
            if p.startswith("s3://"):
                p = p.split("/", 3)[3]                       # drop s3://<bucket>/
            if p.startswith("images/"):                      # open-data release: relative S3 key
                p = os.path.join(path_prefix, p)
            else:
                p = re.sub(r"^.*?/images/", path_prefix.rstrip("/") + "/images/", p)
        views[k][str(r["ViewPosition"]).upper()] = views[k][str(r["ViewPosition"]).upper()] or p
    rows = defaultdict(list)
    no_side = Counter()
    for _, r in cl.iterrows():
        side = "" if C.is_missing(r.get("side")) else str(r.get("side")).upper()[:1]
        if side not in ("L", "R", "B"):
            # a row without side applies to both breasts only for a negative exam without findings
            has_finding = any(not C.is_missing(r.get(c)) for c in ("massshape", "massmargin", "calcfind", "calcdistri"))
            if str(r.get("asses", "")).strip().upper()[:1] == "N" and not has_finding:
                side = "B"; no_side["negative->both"] += 1
            else:
                no_side[f"skipped asses={str(r.get('asses', '')).strip()[:1]}"] += 1
                continue
        sides = ["L", "R"] if side == "B" else [side]
        for s in sides:
            rows[(r["empi_anon"], r["acc_anon"], s)].append(r)
    recs = []
    for k, rs in rows.items():
        if k not in views:
            continue
        lesions = [_embed_lesion(r) for r in rs]
        v = _merge_lesions(lesions)
        v["density"] = next((l["density"] for l in lesions if l.get("density")), None)
        pid = str(k[0])
        vtag = holdout_tag(k[0])
        split = f"test_embed_{vtag}" if vtag else hsplit(pid, val_frac=0.1, test_frac=0.15)
        first = rs[0]
        recs.append(finalize({
            "sample_id": f"embed:{k[0]}:{k[1]}:{k[2]}", "dataset": "embed", "patient_id": pid,
            "study_id": str(k[1]), "side": k[2], "split": split,
            "views": views[k], "values": v, "report": None,
            "embed_meta": {
                "vendor": exam_vendor.get(k[1], "unknown"), "vendor_holdout": vtag,
                "exam_type": "screening" if re.search("screen", str(first.get("desc", "")), re.I) else
                             ("diagnostic" if not C.is_missing(first.get("desc")) else None),
                "cohort_num": None if C.is_missing(first.get("cohort_num")) else str(first.get("cohort_num")),
                "loc_num": None if C.is_missing(first.get("loc_num")) else str(first.get("loc_num")),
                "race": race_by_patient.get(k[0]),
            }}))
    write(recs, out)
    vc = Counter(r["embed_meta"]["vendor_holdout"] or "train_pool" for r in recs)
    print("  rows without side:", dict(no_side), "| vendor hold-out breasts:", dict(vc))
    return recs


# ------------------------------------------------------------------ EMBED temporal (paper 2)
CHANGE_CLASSES = ("stable", "decreased", "increased", "resolved")
_CHANGE_PRIORITY = ["stable", "decreased", "resolved", "increased"]   # merge/tie-break order: increased wins

# Raw per-finding `changed` code -> 4-way class. VERIFY (judgment call, 2026-09-25): 'C'
# ("calcifications more coarse, consistent with a benign process") is read as a decrease in
# suspicion rather than a size change and folded into 'decreased'; it is rare in the open-data
# release (n=30) and could instead be split into its own 'benign_evolution' class later.
_CHANGE_CODE_MAP = {
    "N": "stable",
    "+": "increased", "I": "increased", "M": "increased", "O": "increased",
    "-": "decreased", "D": "decreased", "U": "decreased", "X": "decreased", "P": "decreased", "C": "decreased",
    "S": "resolved", "G": "resolved", "R": "resolved",
}


def _decode_changed(raw):
    """'changed' cell (possibly comma-separated, e.g. 'X,U') -> the most severe of
    _CHANGE_PRIORITY across its mapped codes; codes with no entry in _CHANGE_CODE_MAP are ignored."""
    if C.is_missing(raw):
        return None
    vals = [_CHANGE_CODE_MAP[c1.strip()] for c1 in str(raw).split(",") if c1.strip() in _CHANGE_CODE_MAP]
    return max(vals, key=_CHANGE_PRIORITY.index) if vals else None


def _finding_kind(r) -> str:
    """Coarse mass-vs-calc kind used ONLY to disambiguate prior-exam linkage candidates (NOT for
    head attribution, which reuses _embed_lesion). 'mass' also covers EMBED's asymmetry/distortion
    codes (recorded in the same massshape field) -- kept coarse because the cascade below was
    validated at this granularity: 90.5% of directional-change rows resolve to a unique prior
    finding (see embed_linkage_verification.md, 2026-09-25)."""
    if not C.is_missing(r.get("massshape")) or not C.is_missing(r.get("massmargin")):
        return "mass"
    if not C.is_missing(r.get("calcfind")) or not C.is_missing(r.get("calcdistri")):
        return "calc"
    return "other"


def _link_prior(row, prior_rows: list) -> str:
    """Validated cascade (2026-09-25): side+location+kind -> side+kind -> side-only, accepting
    only a UNIQUE candidate at each stage. Returns a status string; the match itself isn't needed
    by the caller (image pairing is at breast level, not per-finding) but the status drives the
    'new' confidence tag."""
    side = row.get("side")
    same_side = [p for p in prior_rows if p.get("side") == side]
    if not same_side:
        return "no_side_match"
    loc, kind = row.get("location"), _finding_kind(row)
    if not C.is_missing(loc):
        cand = [p for p in same_side if p.get("location") == loc and _finding_kind(p) == kind]
        if len(cand) == 1:
            return "side+loc+kind"
        if len(cand) > 1:
            return "ambiguous"
    cand = [p for p in same_side if _finding_kind(p) == kind]
    if len(cand) == 1:
        return "side+kind"
    if len(cand) > 1:
        return "ambiguous"
    return "side_only" if len(same_side) == 1 else "ambiguous"


def embed_temporal(root: str, out: str, clinical_csv=None, metadata_csv=None, legend_csv=None,
                    clinical_reduced_csv=None, path_prefix: str | None = None) -> list:
    """EMBED open data, PAIRED across a patient's consecutive in-sample exams (paper 2).

    One record per (patient, current exam, side) where BOTH the current exam and the
    immediately-preceding in-sample exam (by study_date_anon, one exam-rank apart -- the cascade
    below is validated one step back only) resolve to a breast with a real CC/MLO image, using
    the same clinical<->metadata join as `embed()`. legend_csv/clinical_reduced_csv are accepted
    for call-signature parity with `embed()` but unused here (codes are mapped directly, as in
    `embed()`; race/vendor-holdout are not part of this manifest -- see split note below).

    Per finding-row change is decoded from `changed` (see _decode_changed) and, when absent,
    the row is linked to the prior exam via `_link_prior` to produce a SILVER 'new' candidate tag.
    Both are merged to breast level per PRESENCE head (mass/calc/asymmetry/distortion) using the
    same most-severe convention as `_merge_lesions`:
      change[head]        in CHANGE_CLASSES, or None -- only when >=1 current finding under that
                          head carries an explicit radiologist `changed` code.
      new_candidate[head] in {"strong", "weak", None} -- for findings with NO explicit code:
                          "strong" when the prior exam has no finding at all on that side
                          (~30%% of the uncoded-localized-finding pool in the 2026-09 release),
                          "weak" when the prior exam has same-side findings but none coded
                          (ambiguous -- treat as unlabelled, not as a negative, when training).
    `values`/`prior_values` (+ `prior_labels`) carry the breast-level descriptor STATE (same
    schema as `embed()`, current and prior) so the change label is checkable against the raw
    state delta, not the only source of truth for what the finding looks like.

    Split policy mirrors `embed()` exactly (same vendor hold-out + hsplit salt/fractions) so a
    patient lands in the same split family in both manifests -- this deliberately trades away
    an independent split for paper 2 in order to avoid train/test leakage if the two manifests
    are ever pooled or compared for the same patient.

    NOT every changed-positive finding-row makes it into this manifest: ~23%% of them (see
    embed_linkage_verification.md) reference a prior exam that is itself outside this 20%%
    open-data release and has no images to pair -- those rows are silently absent here, which is
    a ceiling of the public data, not a bug in this builder.
    """
    cl = pd.read_csv(clinical_csv or glob.glob(os.path.join(root, "**", "*clinical*.csv"), recursive=True)[0], low_memory=False)
    md = pd.read_csv(metadata_csv or glob.glob(os.path.join(root, "**", "*metadata*.csv"), recursive=True)[0], low_memory=False)
    cl["study_date_anon"] = pd.to_datetime(cl["study_date_anon"], errors="coerce")

    # ---- exam order per patient -> prior acc_anon one exam-rank back, in-sample only ----
    ed = cl.groupby(["empi_anon", "acc_anon"])["study_date_anon"].first().reset_index()
    ed = ed.sort_values(["empi_anon", "study_date_anon"])
    acc_prior = {}
    for pid, g in ed.groupby("empi_anon"):
        accs = g["acc_anon"].tolist()
        for i, a in enumerate(accs):
            acc_prior[a] = accs[i - 1] if i > 0 else None

    # ---- vendor hold-out, identical to embed() (kept for split parity, see docstring) ----
    mdv = md.copy()
    if "FinalImageType" in mdv.columns:
        mdv = mdv[mdv["FinalImageType"].astype(str).str.upper() == "2D"]
    if "spot_mag" in mdv.columns:
        mdv = mdv[mdv["spot_mag"].isna()]
    exam_vendor = mdv.groupby("acc_anon")["Manufacturer"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iloc[0]).map(_vendor_tag)
    pv = mdv[["empi_anon", "acc_anon"]].drop_duplicates()
    pv["vendor"] = pv["acc_anon"].map(exam_vendor)
    patient_vendors = pv.groupby("empi_anon")["vendor"].agg(set).to_dict()

    def holdout_tag(empi_anon):
        vs = patient_vendors.get(empi_anon, set())
        for v in _EMBED_VENDOR_HOLDOUT:
            if v in vs:
                return v
        return None

    # ---- image views per (empi_anon, acc_anon, side), identical join to embed() ----
    if "FinalImageType" in md.columns:
        md = md[md["FinalImageType"].astype(str).str.upper() == "2D"]
    md = md[md["ViewPosition"].astype(str).str.upper().isin(["CC", "MLO"])]
    if "spot_mag" in md.columns:
        md = md[md["spot_mag"].isna()]
    lat_col = "ImageLateralityFinal" if "ImageLateralityFinal" in md.columns else "ImageLaterality"
    views = defaultdict(lambda: {"CC": None, "MLO": None})
    for _, r in md.iterrows():
        k = (r["empi_anon"], r["acc_anon"], str(r[lat_col]).upper()[:1])
        p = str(r["anon_dicom_path"])
        if path_prefix:
            if p.startswith("s3://"):
                p = p.split("/", 3)[3]
            if p.startswith("images/"):
                p = os.path.join(path_prefix, p)
            else:
                p = re.sub(r"^.*?/images/", path_prefix.rstrip("/") + "/images/", p)
        views[k][str(r["ViewPosition"]).upper()] = views[k][str(r["ViewPosition"]).upper()] or p

    # ---- group clinical rows by (empi_anon, acc_anon, side); same side='B'->L+R rule as embed() ----
    rows = defaultdict(list)
    for _, r in cl.iterrows():
        side = "" if C.is_missing(r.get("side")) else str(r.get("side")).upper()[:1]
        if side not in ("L", "R", "B"):
            has_finding = any(not C.is_missing(r.get(c1)) for c1 in ("massshape", "massmargin", "calcfind", "calcdistri"))
            if str(r.get("asses", "")).strip().upper()[:1] == "N" and not has_finding:
                side = "B"
            else:
                continue
        for s in (["L", "R"] if side == "B" else [side]):
            rows[(r["empi_anon"], r["acc_anon"], s)].append(r)

    recs, link_stats, no_pair = [], Counter(), Counter()
    for (pid, acc, side), rs in rows.items():
        prior_acc = acc_prior.get(acc)
        if prior_acc is None:
            no_pair["no_in_sample_prior"] += 1
            continue
        cur_key, prior_key = (pid, acc, side), (pid, prior_acc, side)
        if cur_key not in views or prior_key not in views:
            no_pair["no_image_pair"] += 1
            continue
        prior_rows = rows.get(prior_key, [])

        cur_lesions = [_embed_lesion(r) for r in rs]
        change = {h: None for h in C.PRESENCE_HEADS}
        new_cand = {h: None for h in C.PRESENCE_HEADS}
        for r, l in zip(rs, cur_lesions):
            heads_here = [h for h in C.PRESENCE_HEADS if l.get(h) == "present"]
            if not heads_here:
                continue
            raw_change = _decode_changed(r.get("changed"))
            if raw_change is not None:
                link_stats["coded"] += 1
                for h in heads_here:
                    change[h] = raw_change if change[h] is None else max((change[h], raw_change), key=_CHANGE_PRIORITY.index)
                continue
            status = _link_prior(r, prior_rows)
            link_stats[status] += 1
            tag = "strong" if status == "no_side_match" else "weak"
            for h in heads_here:
                if new_cand[h] is None or (new_cand[h] == "weak" and tag == "strong"):
                    new_cand[h] = tag

        prior_lesions = [_embed_lesion(r) for r in prior_rows]
        cur_v, prior_v = _merge_lesions(cur_lesions), _merge_lesions(prior_lesions)
        cur_v["density"] = next((l["density"] for l in cur_lesions if l.get("density")), None)
        prior_v["density"] = next((l["density"] for l in prior_lesions if l.get("density")), None)
        prior_labels = C.encode(prior_v)

        pid_s = str(pid)
        vtag = holdout_tag(pid)
        split = f"test_embed_{vtag}" if vtag else hsplit(pid_s, val_frac=0.1, test_frac=0.15)
        recs.append(finalize({
            "sample_id": f"embed_temporal:{pid}:{acc}:{prior_acc}:{side}", "dataset": "embed_temporal",
            "patient_id": pid_s, "study_id": str(acc), "side": side, "split": split,
            "views": views[cur_key], "values": cur_v, "report": None,
            "embed_temporal": {
                "prior_study_id": str(prior_acc), "prior_views": views[prior_key],
                "prior_values": C.decode(prior_labels), "prior_labels": prior_labels,
                "change": change, "new_candidate": new_cand,
            }}))
    write(recs, out)
    print(f"embed_temporal: {len(recs)} paired breasts | dropped: {dict(no_pair)} | finding-linkage outcomes: {dict(link_stats)}")
    return recs


# ------------------------------------------------------------------ INbreast
def inbreast(root: str, out: str) -> list:
    """INbreast (412-row INbreast.xls; Patient ID is 'removed' in public copies, so CC/MLO
    cannot be paired reliably) -> one record per IMAGE, the single view used for both slots.
    Labels: BI-RADS, ACR density (1-4), presence of Mass / Micros / Distortion / Asymmetry ('X')."""
    meta = next(iter(glob.glob(os.path.join(root, "**", "INbreast*.xls*"), recursive=True)), None)
    df = pd.read_excel(meta) if meta else pd.read_csv(glob.glob(os.path.join(root, "**", "INbreast*.csv"), recursive=True)[0], sep=";")
    df.columns = [c.strip() for c in df.columns]
    dcms = {os.path.basename(p).split("_")[0]: p for p in glob.glob(os.path.join(root, "**", "*.dcm"), recursive=True)}
    mark = lambda x: "present" if str(x).strip().upper() == "X" else "absent"
    recs = []
    for _, r in df.iterrows():
        if C.is_missing(r.get("File Name")):
            continue
        fn = str(int(float(r["File Name"])))
        if fn not in dcms:
            continue
        vp = str(r["View"]).upper().strip()
        v = {"birads": C.birads_group(r.get("Bi-Rads")), "density": C.density_letter(r.get("ACR")),
             "mass": mark(r.get("Mass")), "calc": mark(r.get("Micros")),
             "distortion": mark(r.get("Distortion")), "asymmetry": mark(r.get("Asymmetry"))}
        recs.append(finalize({
            "sample_id": f"inbreast:{fn}", "dataset": "inbreast", "patient_id": fn, "study_id": fn,
            "side": str(r["Laterality"]).upper()[:1], "split": "test",
            "views": {"CC": dcms[fn] if vp == "CC" else None, "MLO": dcms[fn] if vp != "CC" else None},
            "values": v, "report": None}))
    write(recs, out)
    return recs


BUILDERS = {"cbis_ddsm": cbis_ddsm, "vindr": vindr, "cmmd": cmmd, "cdd_cesm": cdd_cesm,
            "embed": embed, "embed_temporal": embed_temporal, "inbreast": inbreast}
