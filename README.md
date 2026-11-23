# cbmammo-prior-audit

Code and aggregate results for

> Orazayev Y., Abdikenov B. **Do Temporal Models Read the Prior Exam? An Audit of Interval-Change Supervision in the Open EMBED Mammography Release.** Manuscript under review. Astana IT University, Astana, Kazakhstan.

The study asks whether models that read a current--prior mammogram pair use the prior, using the radiologist `changed` codes of the open 20 % release of the Emory Breast Imaging Dataset (EMBED). It pairs each breast with the same breast in the patient's preceding in-sample exam (70,065 current--prior breast pairs, 4,552 of them with a change code or a silver new-finding label), compares label-free concept differencing with learned cross-attention heads (concept-bottleneck and opaque, three seeds each), and adds prior-usage controls: a different-patient prior, an identity prior, and a current-only retrain with the same parameter count.

## Findings in one paragraph

Only 1.5 % of training pairs carry a change label (3 resolved and 24 changed examples for masses). Differencing the concept predictions of frozen static models does not recover coded change (pooled AUROC 0.48, 95 % CI 0.41--0.55; a gold-label oracle reaches 0.34--0.43). Learned heads are above chance for some tasks on some splits, but heads that never see the prior match or exceed them; the prior adds 0.02--0.05 AUROC only for a silver new-finding label that is defined by the absence of a prior finding. Concept-bottleneck and opaque heads do not differ.

## What is in this repository

| Path | Content |
|---|---|
| `cbmammo/` | static concept-bottleneck / opaque models and training (`model.py`, `train_stage1.py`, ...), dataset builders (`cbmammo/prepare`, including `embed_temporal`, the current--prior pair builder; see "How the labels are built" below for what the finding-linkage rule is used for), and the temporal extension: `temporal_concepts.py`, `temporal_data.py`, `temporal_model.py` (cross-attention block, `prior_mode: self` current-only control, loss), `train_stage1_temporal.py` |
| `configs/` | `cb_embed_ft2.yaml`, `opq_embed_ft2.yaml` (frozen static bases), `cb_temporal_lab.yaml`, `opq_temporal_lab.yaml` (temporal heads), `cb_self_temporal_lab.yaml`, `opq_self_temporal_lab.yaml` (current-only controls), `smoke*.yaml` (CPU smoke tests) |
| `audit/` | `label_density.py`, `concept_diff_infer.py`, `concept_diff_analysis.py`, `registration_baseline.py`, `registration_analysis.py`, `temporal_eval.py`, `prior_shuffle_eval.py`, `prior_shuffle_analysis.py` |
| `results/tables/` | aggregate result tables behind every number in the manuscript (no patient-level data) |
| `results/runs/` | `run.json` (exact configuration and best validation score) and `train_log.json` for the 6 frozen bases and 12 temporal runs |
| `scripts/`, `tests/` | synthetic-data generators for smoke tests; unit tests (including the test that the current-only control is independent of the prior) |

Not included: EMBED images and tables, the pair manifest `manifests/embed_temporal.jsonl` (it carries patient and accession identifiers of the EMBED release), model weights, image caches.

## How the labels are built

* **Pairs.** A pair is a breast of a current exam and the same breast in the patient's immediately preceding exam in the open release (`embed_temporal` in `cbmammo/prepare/builders.py`). Breasts without an in-sample prior or without an image pair are dropped.
* **Coded change.** The change label of a finding row is taken directly from the radiologist's `changed` code on the current finding (`_decode_changed`). It does not depend on any matching of findings.
* **Finding-linkage cascade (`_link_prior`).** For a current finding it looks for a single candidate in the prior exam: side, location and kind (mass or calcification), then side and kind, then side alone; several candidates count as ambiguous. In the manifest builder it is applied to uncoded findings only. A finding with no code is tagged a `strong` new candidate when the prior exam has no finding at all on that side, and `weak` otherwise; `weak` candidates get weight 0 in the loss and are not evaluated (`encode_new` in `cbmammo/temporal_concepts.py`).
* **Silver new-finding label.** Class "not new" = the finding carries any explicit change code (including "no significant change"). Class "new" = an uncoded finding with a `strong` tag. The two classes therefore differ both in whether a prior finding exists and in whether a code was recorded.
* **The 90.5 % figure.** The share of directional-code rows with an in-sample prior that resolve to a single prior finding (reported in the manuscript) comes from a separate check that is not part of this repository.
* **Vendor hold-outs.** `test_embed_ge` and `test_embed_fuji` hold every record of each patient who has at least one GE or Fujifilm 2D exam (see the `embed()` docstring), so a record in these splits can still pair two Hologic exams.

## Data

EMBED is distributed through the AWS Open Data programme under its own data-use terms; obtain it from the original source and respect those terms. CBIS-DDSM and VinDr-Mammo are used to train the static bases and are available from their original sources.

## Reproduction

Commands are run from the repository root; `PYTHONPATH=.` may be needed. Steps 2--4 need a GPU.

```bash
pip install -r requirements.txt

# 1. manifests (from your own copies of the data)
python -m cbmammo.prepare cbis_ddsm      --root <CBIS-DDSM root>   --out manifests/cbis_ddsm.jsonl
python -m cbmammo.prepare vindr          --root <VinDr-Mammo root> --out manifests/vindr.jsonl
python -m cbmammo.prepare embed          --root <EMBED root>       --out manifests/embed.jsonl
python -m cbmammo.prepare embed_temporal --root <EMBED root>       --out manifests/embed_temporal.jsonl
# the static-base configs also list cmmd / cdd_cesm / inbreast as monitoring test sets; build them the same way or edit the YAML
python audit/label_density.py            # label yield of the pair manifest (Table 2)

# 2. frozen static bases: 2 arms x 3 seeds  ->  runs/{cb,opq}_embed_ft2_s{0,1,2}
for s in 0 1 2; do
  python -m cbmammo.train_stage1 --config configs/cb_embed_ft2.yaml  --seed $s
  python -m cbmammo.train_stage1 --config configs/opq_embed_ft2.yaml --seed $s
done

# 3. temporal heads and current-only controls (each seed on the matching seed's base)
for s in 0 1 2; do
  for a in cb opq; do
    python -m cbmammo.train_stage1_temporal --config configs/${a}_temporal_lab.yaml      --seed $s --override base_checkpoint=runs/${a}_embed_ft2_s$s/best.pt
    python -m cbmammo.train_stage1_temporal --config configs/${a}_self_temporal_lab.yaml --seed $s --override base_checkpoint=runs/${a}_embed_ft2_s$s/best.pt
  done
done

# 4. label-free concept differencing, evaluation, prior-usage controls (bootstrap B = 1000 as reported)
python audit/concept_diff_infer.py --runs cb_embed_ft2_s0 cb_embed_ft2_s1 cb_embed_ft2_s2 --out cd_out
python audit/concept_diff_analysis.py cd_out results/tables/concept_diff_results.csv 1000
python audit/registration_baseline.py --out reg_out --cache cache/prep      # CPU only, ~10 min; registration + image-subtraction baseline and its two controls
python audit/registration_analysis.py reg_out results/tables/registration_baseline_results.csv 1000
python audit/temporal_eval.py cd_out runs results/tables/temporal_eval_final.csv 1000     # also writes temporal_eval_final_paired.csv
python audit/prior_shuffle_eval.py prior_out cb_temporal_lab_s0 cb_temporal_lab_s1 cb_temporal_lab_s2 opq_temporal_lab_s0 opq_temporal_lab_s1 opq_temporal_lab_s2
python audit/prior_shuffle_analysis.py prior_out cd_out results/tables/prior_shuffle_final.csv 1000
```

### Tables and manuscript

| Manuscript table | File in `results/tables/` |
|---|---|
| 2, label yield | `label_yield.csv`, `label_yield_train_descriptors.csv` (transcribed from the output of `audit/label_density.py`) |
| 3, concept differencing | `concept_diff_results.csv` |
| 4, registration / difference-image baseline | `registration_baseline_results.csv` (all statistics, splits and descriptors), `registration_baseline_results_controls.csv` (different-patient prior and synthetic-lesion controls), `registration_baseline_results_primary.json` (statistic chosen from the synthetic-lesion control) |
| 5, learned heads per split (prior-using heads CB and OPQ, current-only controls CB-c and OPQ-c) | `temporal_eval_final.csv` |
| 6, prior-using minus current-only, CB minus opaque | `temporal_eval_final_paired.csv` |
| 7, true / different-patient / identity prior | `prior_shuffle_final.csv` |

AUROCs are split-stratified averages (weights proportional to n_pos x n_neg) or per-split values; intervals are 95 % patient-cluster bootstrap intervals; paired differences share resamples.

`temporal_eval_final.csv` and `temporal_eval_final_paired.csv` list five methods: `concept_diff (label-free)`, `CB learned (concept heads)`, `opaque learned`, and the current-only controls `CB current-only` and `opaque current-only` (three-seed ensembles). The two files were regenerated on 24 November 2026 from the stored scores with `audit/temporal_eval.py` as it is in this repository (B = 1000). The rows of the first three methods are identical to the previous version of the files; the rows of the two current-only controls and the paired differences that involve them are new. The table numbers above are those of the revised manuscript (Table 1 there is the related-work table).

## Tests and smoke test

```bash
python -m pytest -q tests                       # unit tests, CPU, no downloads (tiny random encoder)
python scripts/make_synthetic.py                # synthetic static data
python scripts/make_synthetic_temporal.py       # synthetic current--prior data
python -m cbmammo.train_stage1 --config configs/smoke.yaml
python -m cbmammo.train_stage1_temporal --config configs/smoke_temporal.yaml        # add _self for the current-only control
```

The smoke test checks that the pipeline runs; its numbers mean nothing.

## Provenance notes

* The audit scripts ran on a GPU server with absolute paths; the paths were made relative to the repository root afterwards and the files were syntax-checked, but the scripts were not re-executed end to end after that edit, with one exception: `audit/temporal_eval.py` was run unchanged on the stored scores when the current-only rows were added (see above). The training code and configs are unchanged apart from `cache_dir`.
* `configs/cb_embed_ft2.yaml` and `opq_embed_ft2.yaml` were reconstructed from the saved `run.json` of the corresponding runs; seeds of one arm differ only in `seed` and the dataloader worker count.
* `cbmammo/prepare/builders.py` is the file used for the runs with the builder for a private clinic dataset removed (not used here). The `cbmammo/` package is shared with the repository of the companion site-shift study (`bcaitu/cbmammo-site-shift`).
* The original full-manifest runs (before the label-aware sampler) collapsed to constant predictions and are not part of the reported results; their configs are omitted.

* The registration baseline was added last. Its design choices (image size, smoothing, the three tail statistics, the control amplitudes and lesion size) were fixed on an 80-pair smoke run using the failure diagnostics and the synthetic-lesion control; the AUROCs of that smoke run against the labels were printed but not used to choose anything. The primary statistic is the one with the highest synthetic-lesion AUROC at amplitude 0.4, computed by `registration_analysis.py`.
* The registration baseline is affine and detects only conspicuous synthetic lesions (AUROC about 0.65 at amplitude 0.4 and 0.78 at 0.6), so its null result does not exclude subtle interval change.
* The diagnostic montage of registered image pairs written by `registration_baseline.py` contains EMBED image content and must not be committed or shared.

## Licence and citation

Apache-2.0 (see `LICENSE`). Please cite the manuscript (see `CITATION.cff`).
