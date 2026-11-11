# Residual CNN–Mamba hybrid for lung cancer histopathology: code, data and results

Code and released data for the paper

> Soni, M. and Sharma, S. Under leakage-controlled evaluation, an explainable
> CNN–Mamba hybrid shows no accuracy gain over its convolutional counterpart in
> lung cancer histopathology. Manuscript; not yet peer reviewed.

The study compares a residual CNN–Mamba hybrid with its own CNN pathway,
EfficientNet-B0, ViT-S/16, Swin-T, a frozen Phikon linear probe and, as a
secondary comparator, VMamba-T, on LC25000 (with copies of one tissue field kept
on one side of every split) and LungHist700 (with each patient on one side),
and measures the hybrid's reliance on its Mamba branch and the faithfulness of
every model's explanations against a random map.

This repository holds what is needed to reproduce that work: the models, the
training and evaluation pipeline, the experiment queues that ran every model,
the recovered LC25000 source groups, the split files, the per-image predictions
and explanation scores of every evaluated model, and the scripts that
regenerate every table and figure of the paper from those predictions, without
a GPU. The datasets themselves are not redistributed.

## What is where

| Part | Where |
|---|---|
| Residual hybrid: Mamba (SS2D) over the CNN's stride-16 feature map, added through a zero-initialised residual, so the model starts as its CNN pathway and the Mamba contribution is measurable | `models/hybrid_res.py` (`hybrid_res`, `hybrid_res_xattn`) |
| Comparators under one training recipe: CNN pathway alone, the original dual-pathway hybrid, EfficientNet-B0, ViT-S/16, Swin-T, a frozen Phikon linear probe, and VMamba-T | `models/baselines.py`, `models/hybrid_model.py`, `models/vmamba_vendor/`, `tools/fm_probe.py` |
| LC25000 source-image recovery (716 groups) and the grouped split; the per-file split as the leaky reference | `tools/group_lc25000.py`, `lc25000_groups.json`, `data/dataset.py` |
| LungHist700 patient-level and image-level five-fold cross-validation, three and seven classes | `tools/prepare_lunghist700.py`, `splits/` |
| External test LC25000 → LungHist700 (20× and 40×) | `scripts/queue_lh700.sh` |
| Pathway attribution (Mamba ablation) and deletion/insertion faithfulness against a random map | `tools/xai_metrics.py` |
| Parameters, FLOPs, latency, memory | `tools/efficiency.py` |
| Cluster-aware statistics (bootstrap and sign-flip tests over source groups or patients, Holm) and every table and figure | `tools/paper_tables.py`, `tools/supp_tables.py`, `tools/paper_figures.py`, `tools/manuscript_tables.py` |
| Per-image predictions, explanation scores, tables and figures | `results/` |

The variant carried forward (`hybrid_res` or `hybrid_res_xattn`) was chosen
only by seed-42 validation accuracy on LC25000, under a rule fixed before
either variant was trained (`scripts/queue_paper1.sh`). It selected
`hybrid_res` (`logs/selected_hybrid.txt`).

## Data

**LC25000** (Borkowski et al., 2019, arXiv:1912.12142), lung subset: 15,000
images in `data/lung_aca`, `data/lung_n`, `data/lung_scc` (`download_dataset.py`,
or the Kaggle mirror `andrewmvd/lung-and-colon-cancer-histopathological-images`).
The images are about 20 augmented copies each of 750 originals, so a per-file
split leaks copies of the same tissue field across train and test.
`lc25000_groups.json` assigns every image to one of 716 recovered source groups
(adjusted Rand index 0.949 against LC25000-clean, Batchkala et al., 2024).

**LungHist700** (Diosdado et al., Sci. Data 11:1088, 2024; CC BY 4.0;
doi:10.6084/m9.figshare.25459174): 691 images from 45 patients. After
downloading and extracting it (`data.csv` plus `images/`):

```bash
python tools/prepare_lunghist700.py --src <extracted> --out ./data_lh700 \
    --splits ./splits/lunghist700
python tools/prepare_lunghist700.py --src <extracted> --granularity sub \
    --out ./data_lh700_7c --splits ./splits/lunghist700_7c
```

The resulting split files are in `splits/` (seed 42), so folds can be reused
without running the script.

**LC25000-clean** (Batchkala et al., 2024), used for the clean-subset
sensitivity analysis: download `lc25000_image_groups.csv` (URL in
`tools/compare_groupings.py`) into the repository root.

## Regenerating the tables and figures (no GPU)

```bash
pip install -r requirements.txt
python tools/paper_tables.py         # results/_paper/tables.md and tables.json
python tools/supp_tables.py          # results/_paper/supp_tables.md and .json
python tools/paper_figures.py        # results/_paper/figures/ (Figs 2-6)
python tools/manuscript_tables.py    # Tables 1 and 2 as printed in the paper
```

These read only the committed per-image predictions and explanation scores.
`python -m pytest -q -m slow` regenerates every table this way and checks that
it matches the committed `results/_paper/` exactly (it needs
`lc25000_image_groups.csv`; without it the check is skipped).
`python -m pytest -q` runs the fast tests.

Figure 1 is drawn by `tools/fig1_design.py`; Supplementary Figs S1–S3 by
`tools/grouping_validation.py` (`--plot_only` redraws S1 from its saved data)
and `tools/figS2_examples.py`, which need the dataset images. They are in
`docs/manuscript/figures/`, with the grouping-validation data behind S1.

## Reproducing the experiments

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

`requirements-lock.txt` pins the versions actually used (Python 3.13.16,
PyTorch 2.14.1, CUDA 13.0, one NVIDIA A10G 24 GB; `docs/AWS_SETUP.md`
describes the cloud instance). Then, from the repository root:

```bash
tmux new-session -d -s queue "scripts/queue_paper1.sh > logs/queue_paper1.log 2>&1"
tmux new-session -d -s xai   "WAIT_FOR=queue scripts/queue_xai.sh > logs/queue_xai.log 2>&1"
PYTHONPATH=~/fm_pkgs python3 tools/fm_probe.py     # Phikon reference (see the lock file)
```

`scripts/queue_paper1.sh` runs, in order: the seed-42 baselines, the two
residual variants, the pre-specified selection, the original hybrid, the
LungHist700 queue (`scripts/queue_lh700.sh`), seeds 1337 and 2024, the
leaky-split runs and the remaining external tests. `scripts/queue_xai.sh` then
runs the cost benchmark, the explanation metrics and the rating kit used for
the example panels. Every step skips finished runs, so either script can be
restarted.

Shared training recipe (`scripts/run_experiment.sh`, `scripts/run_study.sh`):
AdamW, learning rate 1e-4 (1e-5 for pretrained backbones), weight decay 0.05,
3 warm-up epochs then cosine decay over 50, batch 32, label smoothing 0.1,
float16 mixed precision, early stopping on validation loss (patience 5), best
checkpoint by validation accuracy. On LungHist700 that recipe underfitted, and
a small-data protocol was fixed before any further LungHist700 result
(`scripts/queue_lh700.sh`): 100 epochs without early stopping, and a learning
rate for each architecture chosen from {1e-4, 1e-3} on fold 0
(`logs/lh700_lr_selection.txt`). Single runs:

```bash
scripts/run_experiment.sh hybrid_res_d4_224px_s42 --arch hybrid_res --grad_checkpoint
python evaluate.py --checkpoint checkpoints/<run>/best_model.pth --data_root ./data
```

`evaluate.py` rebuilds the architecture, split and seed from the checkpoint, so
it scores the test set that was actually held out. Pass `--split_file` for an
external test.

## Analyses added after the pre-submission review

Their protocol (`docs/added_analyses_protocol.md`) was fixed before any of them
was run, and `scripts/queue_review.sh` runs them:

- **VMamba-T** (`--arch vmamba_tiny`): the authors' implementation, vendored in
  `models/vmamba_vendor/` (MIT; only an optional import changed), configuration
  vssm1_tiny_0230s with their hash-checked ImageNet-1K weights. Its paired
  comparisons with hybrid-res are secondary, outside the Holm family. Its
  selective scan runs on mamba-ssm's CUDA kernel; for PyTorch 2.14 this was
  built from source (mamba-ssm v2.3.2.post1, with
  `-gencode arch=compute_86,code=sm_86` and `-std=c++20` in its `setup.py`),
  and `tools/check_vmamba_kernel.py` compares it with the authors' PyTorch
  reference before any run (`logs/vmamba_kernel_check.txt`).
- **Learning-rate sensitivity** on LungHist700: a third rate, 3e-3, for every
  architecture, with the runs repeated where it wins
  (`logs/lh700_lr_sensitivity.txt`).
- **Random-map granularity**: every faithfulness analysis re-scored against a
  7 × 7 random map (`xai_metrics_rg7/`, `xai_metrics_meandel_rg7/` in the run
  folders).

## Development history

This repository was created for publication and contains only what is needed
to reproduce the paper; the work was done in a private development repository.
Three rules in the paper were fixed before the results they govern existed:
the choice between the two residual variants, the LungHist700 small-data
protocol and the protocol of the added analyses. `docs/development_history.md`
lists the commits that fixed them, with their dates. The dated history of the
development repository is available to editors and reviewers on request.

## Stain normalisation

The study uses **no stain normalisation**; training applies colour jitter, and
robustness to stain differences between sites is measured by the external
test. The `MacenkoNormalize` used until 2026-10-11 swapped haematoxylin and
eosin and mis-scaled concentrations; all runs trained on its output were
discarded. It has been rewritten to follow the reference algorithm, but with
the fixed background intensity (Io = 240) it still fails on LungHist700, whose
background is darker, so it is not used.

## Limitations

The paper's Discussion gives the full list. In short: both datasets are
single-centre, and LungHist700 has 45 patients, so no significant difference
does not mean equivalence; images are classified at 224 × 224 without
whole-slide context; the LC25000 grouping is automatic; explanation
faithfulness depends on the perturbation used, and no expert rated the
explanations; each LungHist700 fold was trained with one seed and all trained
models shared one recipe; the hybrids' latency reflects a pure-PyTorch
selective scan; VMamba-T and the two sensitivity analyses were added after the
main results were known.

## Layout

```
config.py, train.py, evaluate.py      configuration, training loop, evaluation
compare_models.py                     McNemar and bootstrap helpers
aggregate_results.py                  multi-seed summary (used by scripts/run_study.sh)
download_dataset.py                   LC25000 download
lc25000_groups.json                   recovered LC25000 source groups
data/      dataset.py (grouped, per-file and explicit splits), transforms.py
models/    hybrid_res.py, hybrid_model.py, cnn_pathway.py, mamba_pathway.py,
           ssm_block.py (selective scan, SS2D), fusion_module.py,
           baselines.py, xai.py (Grad-CAM++), vmamba_vendor/ (VMamba, MIT)
scripts/   experiment queues and runners (each changes to the repository root)
tools/     data preparation, grouping, explanation metrics, cost, statistics,
           tables and figures
tests/     pytest suite (add -m slow for the table regeneration check)
splits/    LungHist700 fold and external-test split files
results/   per-run metrics, per-image predictions and explanation scores;
           _paper/ (tables and figures), _efficiency/ (cost)
logs/      the selection and kernel-check records cited in the paper
docs/      added_analyses_protocol.md, development_history.md, AWS_SETUP.md;
           manuscript/figures/ (Fig. 1 and Supplementary Figs S1-S3)
```

## Licence and citation

The code is released under the MIT licence (`LICENSE`). Third-party material
keeps its own terms:
- VMamba (`models/vmamba_vendor/`, from github.com/MzeroMiko/VMamba) is under
  the MIT licence, Copyright (c) 2024 MzeroMiko (`models/vmamba_vendor/LICENSE`);
  mamba-ssm, which provides its CUDA kernel, is a dependency and is not
  included;
- Phikon, used frozen by `tools/fm_probe.py`, is under the Owkin non-commercial
  licence; its weights are not included;
- the datasets are not redistributed here; the example panels in
  `docs/manuscript/figures/figS2_examples.pdf` and `figS3_examples_lh700.pdf`
  show images from LC25000 and LungHist700, which remain under the datasets'
  own terms (LungHist700: CC BY 4.0).

If you use this code or data, please cite the paper and this release
(`CITATION.cff`; the release is archived on Zenodo), together with LC25000
(Borkowski et al., 2019), LungHist700 (Diosdado et al., 2024), LC25000-clean
(Batchkala et al., 2024) and Phikon (Filiot et al., 2023) where you use them.
