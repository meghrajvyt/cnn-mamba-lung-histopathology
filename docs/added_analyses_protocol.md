# Analyses added after the pre-submission review

Fixed on 2026-10-20, before any of the analyses below was run, and committed
to this repository before the first run started. The analyses of the
submitted manuscript are unchanged. Every analysis here is reported whatever
it shows. Methods states that they were added after the main results were
known, following a pre-submission review, and refers to this file.

The review raised three points:

1. No standalone state-space classifier was compared.
2. On LungHist700 every architecture chose the larger of the two learning
   rates tried (10⁻⁴ and 10⁻³), so better settings may lie above that range.
3. The faithfulness analysis compares every saliency map with a 14 × 14
   random map. The maps of the CNN pathway, EfficientNet-B0 and Swin-T are
   7 × 7, so their spatial granularity differs from the baseline's.

`scripts/queue_review.sh` runs all three analyses.

## A1. Standalone state-space comparator: VMamba-T

**Model.**
- VMamba-T (Liu et al., NeurIPS 2024), configuration `vssm1_tiny_0230s`.
- Weights: the authors' ImageNet-1K checkpoint (`vssm1_tiny_0230s_ckpt_epoch_264.pth`, release `#v2cls` of github.com/MzeroMiko/VMamba).
- Code: the authors' single-file implementation (`vmamba.py`, MIT licence), vendored unchanged except for imports.
- Classifier: the ImageNet classification layer is replaced by a new linear layer for the task, preceded by dropout 0.3, as for the timm comparators.
- Stochastic depth: the implementation's default, as the timm comparators keep theirs.
- Learning rates: pretrained parameters train at 0.1 × the learning rate, the new layer at 1 ×, as for every pretrained comparator.

**Selective scan.**
- The authors' fused CUDA kernel is used.
- Before any training run, its outputs and gradients are compared with the authors' pure-PyTorch reference on random inputs. The maximum absolute differences are recorded in `logs/vmamba_kernel_check.txt`.
- If the kernel cannot be built with the environment used for the other models, VMamba-T runs in a separate environment, and its versions are recorded.

**Training.** Identical to the other comparators.
- LC25000: grouped split, seeds 42, 1337 and 2024; the shared recipe (learning rate 10⁻⁴, backbone 10⁻⁵, 50-epoch budget, early stopping with patience 5, batch 32), through `scripts/run_study.sh`.
- LungHist700: the small-data protocol.
  - 100 epochs; the best-validation checkpoint is evaluated.
  - The learning rate is chosen from {10⁻⁴, 10⁻³} by fold-0 validation accuracy on the three-class patient-level split; ties keep 10⁻⁴.
  - The chosen rate is used for every LungHist700 run of the model: three-class and seven-class patient-level folds.
  - Record: `logs/lh700_lr_selection_vmamba.txt`.
- External test: its LC25000 checkpoints are scored on LungHist700 at 20× and 40×, as for the other models.

**Analyses.**
- The metrics of Tables 1 and 2.
- Hybrid-res against VMamba-T, as a paired comparison with clusters as the unit (bootstrap 95% CI and sign-flip test). It is reported as a secondary comparison outside the pre-specified family of five Holm-adjusted comparisons, which is unchanged.
- Cost: parameters, FLOPs (a lower bound, as for the other Mamba-based models), latency, throughput and memory.

**Not run for VMamba-T:** leaky splits, explanation faithfulness and pathway attribution.

## A2. Learning-rate sensitivity on LungHist700

- **Runs.** One extra fold-0 selection run at 3 × 10⁻³ (backbone 3 × 10⁻⁴), with otherwise the same protocol, for each of the six architectures: hybrid-res, CNN pathway, EfficientNet-B0, ViT-S, Swin-T and VMamba-T.
- **Rule.** For each architecture, the sensitivity rate is the one with the highest fold-0 validation accuracy among {10⁻⁴, 10⁻³, 3 × 10⁻³}; ties keep the smaller rate.
- **If the sensitivity rate is 3 × 10⁻³:** all of that architecture's LungHist700 patient-level runs (three-class and seven-class, folds 0–4) are repeated at that rate. They are reported as a sensitivity analysis next to the primary results.
- **Primary results:** those of the pre-specified protocol, unchanged.
- **If 3 × 10⁻³ is chosen:** the paper says that the optimum may lie higher still.
- **Record:** `logs/lh700_lr_sensitivity.txt`.

## A3. Random-baseline granularity

- **Runs.** Every faithfulness analysis of the paper is repeated with a 7 × 7 random grid (`--random_grid 7`, `--skip_attribution`):
  - LC25000: every non-leaky 224-pixel run;
  - LungHist700: every three-class patient-level fold model.
- **Kept the same:** both deletion canvases, the same images, the same random seed and the same number of steps.
- **Output:** `xai_metrics_rg7/` and `xai_metrics_meandel_rg7/` in each run's results folder.
- **Reported:** each map's margin over the 7 × 7 random map, next to its margin over the 14 × 14 random map of the primary analysis, as a sensitivity analysis.
