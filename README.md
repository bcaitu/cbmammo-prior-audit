# cbmammo-prior-audit

Code and aggregate results for

> Orazayev Y., Abdikenov B. **Do Temporal Models Read the Prior Exam? An Audit of Interval-Change Supervision in the Open EMBED Mammography Release.** Manuscript in preparation. Astana IT University, Astana, Kazakhstan.

The study asks whether models that read a current--prior mammogram pair use the prior, using the radiologist `changed` codes of the open 20 % release of the Emory Breast Imaging Dataset (EMBED). It links 70,065 current--prior breast pairs (4,552 with a change code or a silver new-finding label), compares label-free concept differencing with learned cross-attention heads (concept-bottleneck and opaque, three seeds each), and adds prior-usage controls: a different-patient prior, an identity prior, and a current-only retrain with the same parameter count.

## Findings in one paragraph

Only 1.5 % of training pairs carry a change label (3 resolved and 24 changed examples for masses). Differencing the concept predictions of frozen static models does not recover coded change (pooled AUROC 0.48, 95 % CI 0.41--0.55; a gold-label oracle reaches 0.34--0.43). Learned heads are above chance for some tasks on some splits, but heads that never see the prior match or exceed them; the prior adds 0.02--0.05 AUROC only for a silver new-finding label that is defined by the absence of a prior finding. Concept-bottleneck and opaque heads do not differ.

## What is in this repository

| Path | Content |
|---|---|
| `cbmammo/` | static concept-bottleneck / opaque models and training (`model.py`, `train_stage1.py`, ...), dataset builders (`cbmammo/prepare`, including `embed_temporal`, the current--prior pair builder with the finding-linkage rule), and the temporal extension: `temporal_concepts.py`, `temporal_data.py`, `temporal_model.py` (cross-attention block, `prior_mode: self` current-only control, loss), `train_stage1_temporal.py` |
| `configs/` | `cb_embed_ft2.yaml`, `opq_embed_ft2.yaml` (frozen static bases), `cb_temporal_lab.yaml`, `opq_temporal_lab.yaml` (temporal heads), `cb_self_temporal_lab.yaml`, `opq_self_temporal_lab.yaml` (current-only controls), `smoke*.yaml` (CPU smoke tests) |
| `audit/` | `label_density.py`, `concept_diff_infer.py`, `concept_diff_analysis.py`, `temporal_eval.py`, `prior_shuffle_eval.py`, `prior_shuffle_analysis.py` |
| `results/tables/` | aggregate result tables behind every number in the manuscript (no patient-level data) |
| `results/runs/` | `run.json` (exact configuration and best validation score) and `train_log.json` for the 6 frozen bases and 12 temporal runs |
| `scripts/`, `tests/` | synthetic-data generators for smoke tests; unit tests (including the test that the current-only control is independent of the prior) |

Not included: EMBED images and tables, the pair manifest `manifests/embed_temporal.jsonl` (it carries patient and accession identifiers of the EMBED release), model weights, image caches.

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
python audit/label_density.py            # label yield of the pair manifest (Table 1)

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
python audit/temporal_eval.py cd_out runs results/tables/temporal_eval_final.csv 1000     # also writes temporal_eval_final_paired.csv
python audit/prior_shuffle_eval.py prior_out cb_temporal_lab_s0 cb_temporal_lab_s1 cb_temporal_lab_s2 opq_temporal_lab_s0 opq_temporal_lab_s1 opq_temporal_lab_s2
python audit/prior_shuffle_analysis.py prior_out cd_out results/tables/prior_shuffle_final.csv 1000
```

### Tables and manuscript

| Manuscript table | File in `results/tables/` |
|---|---|
| 1, label yield | `label_yield.csv`, `label_yield_train_descriptors.csv` (transcribed from the output of `audit/label_density.py`) |
| 2, concept differencing | `concept_diff_results.csv` |
| 3, learned heads per split | `temporal_eval_final.csv` |
| 4, prior-using minus current-only, CB minus opaque | `temporal_eval_final_paired.csv` |
| 5, true / different-patient / identity prior | `prior_shuffle_final.csv` |

AUROCs are split-stratified averages (weights proportional to n_pos x n_neg) or per-split values; intervals are 95 % patient-cluster bootstrap intervals; paired differences share resamples.

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

* The audit scripts ran on a GPU server with absolute paths; the paths were made relative to the repository root afterwards and the files were syntax-checked, but the scripts were not re-executed end to end after that edit. The training code and configs are unchanged apart from `cache_dir`.
* `configs/cb_embed_ft2.yaml` and `opq_embed_ft2.yaml` were reconstructed from the saved `run.json` of the corresponding runs; seeds of one arm differ only in `seed` and the dataloader worker count.
* `cbmammo/prepare/builders.py` is the file used for the runs with the builder for a private clinic dataset removed (not used here). The `cbmammo/` package is shared with the repository of the companion site-shift study (`bcaitu/cbmammo-site-shift`).
* The original full-manifest runs (before the label-aware sampler) collapsed to constant predictions and are not part of the reported results; their configs are omitted.

## Licence and citation

Apache-2.0 (see `LICENSE`). Please cite the manuscript (see `CITATION.cff`).
