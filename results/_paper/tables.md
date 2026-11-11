# Paper 1 results (selected hybrid: Hybrid-res)

## T1. LC25000, leakage-controlled split (test size varies with the seed's group partition)
Mean (minimum–maximum) over seeds, in %.

| Model | Seeds | Acc | BalAcc | F1 | AUC | MCC |
|---|---|---|---|---|---|---|
| Hybrid-res (selected) | 3 | 97.15 (95.63–98.10) | 97.23 (95.76–98.07) | 97.19 (95.63–98.11) | 99.76 (99.62–99.92) | 95.76 (93.52–97.19) |
| Hybrid-res-xattn | 1 | 99.34 | 99.34 | 99.34 | 99.97 | 99.00 |
| Original hybrid | 1 | 98.94 | 98.95 | 98.95 | 99.95 | 98.41 |
| CNN pathway | 3 | 97.43 (95.63–99.16) | 97.50 (95.64–99.17) | 97.48 (95.63–99.17) | 99.66 (99.50–99.96) | 96.15 (93.43–98.74) |
| EfficientNet-B0 | 3 | 96.81 (95.40–97.83) | 96.87 (95.40–97.81) | 96.87 (95.42–97.85) | 99.45 (99.11–99.91) | 95.23 (93.09–96.78) |
| ViT-S/16 | 3 | 96.95 (94.85–99.29) | 97.04 (94.94–99.31) | 97.01 (94.86–99.30) | 99.43 (98.93–99.99) | 95.42 (92.31–98.94) |
| Swin-T | 3 | 97.40 (95.72–98.63) | 97.47 (95.81–98.62) | 97.44 (95.72–98.64) | 99.44 (98.88–99.88) | 96.11 (93.61–97.95) |
| Phikon linear probe | 3 | 99.19 (98.91–99.36) | 99.21 (98.90–99.41) | 99.20 (98.91–99.39) | 99.98 (99.97–100.00) | 98.78 (98.36–99.03) |
| VMamba-T | 3 | 96.53 (94.62–98.36) | 96.66 (94.81–98.34) | 96.58 (94.62–98.38) | 99.40 (99.06–99.93) | 94.85 (92.08–97.57) |

## T2. LC25000 paired tests: Hybrid-res vs baselines
Δ = hybrid − baseline accuracy (points). Primary inference treats the source-image group as the unit (test images are ~20 augmented copies of ~107 tissue fields): group bootstrap 95% CI and group sign-flip randomisation p, Holm-adjusted across baselines within each seed. Image-level McNemar (discordant = only-hybrid / only-baseline correct) is shown for reference; it treats siblings as independent and so overstates significance. 'secondary': VMamba-T, added after the pre-submission review, is compared outside the Holm family (p unadjusted).

| Seed | Baseline | Δ acc | Group 95% CI | Group p | Holm p | Discordant | McNemar p |
|---|---|---|---|---|---|---|---|
| 42 | CNN pathway | -1.06 | [-3.25, +0.61] | 0.358 | 1.000 | 13 / 37 | 0.001 |
| 42 | EfficientNet-B0 | +0.27 | [-1.56, +1.83] | 0.810 | 1.000 | 34 / 28 | 0.525 |
| 42 | ViT-S/16 | -1.19 | [-3.43, +0.64] | 0.309 | 1.000 | 14 / 41 | <0.001 |
| 42 | Swin-T | -0.53 | [-1.74, +0.42] | 0.450 | 1.000 | 15 / 27 | 0.090 |
| 42 | Phikon linear probe | -1.19 | [-3.61, +1.08] | 0.366 | 1.000 | 16 / 43 | <0.001 |
| 42 | VMamba-T | -0.27 | [-1.84, +1.56] | 0.796 | secondary | 24 / 30 | 0.496 |
| 1337 | CNN pathway | +0.00 | [-1.76, +1.92] | 1.000 | 1.000 | 44 / 44 | 0.915 |
| 1337 | EfficientNet-B0 | +0.23 | [-1.66, +2.19] | 0.857 | 1.000 | 51 / 46 | 0.685 |
| 1337 | ViT-S/16 | +0.77 | [-1.81, +3.52] | 0.600 | 1.000 | 70 / 53 | 0.149 |
| 1337 | Swin-T | -0.09 | [-2.67, +2.89] | 0.976 | 1.000 | 62 / 64 | 0.929 |
| 1337 | Phikon linear probe | -3.28 | [-6.09, -0.62] | 0.018 | 0.089 | 20 / 92 | <0.001 |
| 1337 | VMamba-T | +1.00 | [-2.15, +4.56] | 0.587 | secondary | 80 / 58 | 0.074 |
| 2024 | CNN pathway | +0.21 | [-1.47, +1.81] | 0.844 | 1.000 | 33 / 28 | 0.609 |
| 2024 | EfficientNet-B0 | +0.52 | [-0.65, +1.78] | 0.465 | 1.000 | 31 / 19 | 0.120 |
| 2024 | ViT-S/16 | +1.03 | [-1.13, +3.54] | 0.436 | 1.000 | 49 / 25 | 0.008 |
| 2024 | Swin-T | -0.13 | [-1.72, +1.38] | 0.913 | 1.000 | 22 / 25 | 0.770 |
| 2024 | Phikon linear probe | -1.63 | [-3.18, -0.39] | 0.018 | 0.088 | 3 / 41 | <0.001 |
| 2024 | VMamba-T | +1.12 | [-0.92, +3.61] | 0.430 | secondary | 54 / 28 | 0.006 |

## T3. Leakage inflation (seed 42 for LC25000; pooled CV for LungHist700)
Inflation = leaky − leakage-controlled accuracy (points). LC25000: the two protocols have different test sets, so the CI resamples source groups independently in each. LungHist700: both protocols predict the same images, so the CI and sign-flip p are paired and resample patients.

| Dataset | Model | Controlled | Leaky | Inflation | 95% CI | p |
|---|---|---|---|---|---|---|
| LC25000 (per-file vs grouped) | Hybrid-res | 98.10 | 99.73 | +1.64 | [+0.09, +3.79] | — |
| LC25000 (per-file vs grouped) | CNN pathway | 99.16 | 99.91 | +0.75 | [+0.05, +1.73] | — |
| LC25000 (per-file vs grouped) | EfficientNet-B0 | 97.83 | 99.38 | +1.55 | [+0.19, +3.19] | — |
| LC25000 (per-file vs grouped) | ViT-S/16 | 99.29 | 99.82 | +0.53 | [-0.18, +1.60] | — |
| LC25000 (per-file vs grouped) | Swin-T | 98.63 | 100.00 | +1.37 | [+0.27, +2.97] | — |
| LC25000 (per-file vs grouped) | Phikon linear probe | 99.29 | 100.00 | +0.71 | [+0.00, +2.15] | — |
| LungHist700 all (image- vs patient-level) | Hybrid-res | 73.08 | 88.13 | +15.05 | [+9.44, +20.55] | <0.001 |
| LungHist700 all (image- vs patient-level) | CNN pathway | 73.52 | 88.86 | +15.34 | [+10.45, +20.62] | <0.001 |
| LungHist700 all (image- vs patient-level) | EfficientNet-B0 | 63.10 | 76.99 | +13.89 | [+9.25, +18.79] | <0.001 |
| LungHist700 all (image- vs patient-level) | ViT-S/16 | 76.12 | 89.29 | +13.17 | [+8.15, +18.89] | <0.001 |
| LungHist700 all (image- vs patient-level) | Swin-T | 75.98 | 92.33 | +16.35 | [+9.96, +23.05] | <0.001 |
| LungHist700 all (image- vs patient-level) | Phikon linear probe | 80.46 | 92.76 | +12.30 | [+7.92, +17.01] | <0.001 |

## T4. LungHist700 3-class, patient-level 5-fold CV (all)
Pooled out-of-fold predictions (each image predicted once by a model that never saw its patient), in %. CIs resample patients.

| Model | N | Acc | BalAcc | F1 | AUC | MCC | Acc 95% CI (patient) | Fold acc mean (range) | Acc excl. LR-selection patients |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid-res | 691 | 73.08 | 74.54 | 74.18 | 88.16 | 58.57 | [67.2, 79.0] | 73.12 (69.72–78.83) | 73.65 |
| CNN pathway | 691 | 73.52 | 75.04 | 74.56 | 88.11 | 59.29 | [67.4, 79.4] | 73.53 (71.13–75.91) | 73.10 |
| EfficientNet-B0 | 691 | 63.10 | 65.37 | 63.66 | 80.91 | 44.19 | [56.9, 68.8] | 63.10 (61.31–64.75) | 63.36 |
| ViT-S/16 | 691 | 76.12 | 77.74 | 77.35 | 89.74 | 63.19 | [69.8, 82.0] | 76.18 (70.07–86.13) | 77.26 |
| Swin-T | 691 | 75.98 | 76.75 | 76.94 | 86.51 | 62.80 | [69.2, 82.7] | 75.98 (73.72–80.29) | 75.63 |
| Phikon linear probe | 691 | 80.46 | 81.87 | 81.49 | 92.04 | 69.98 | [75.0, 85.7] | 80.48 (77.70–83.82) | 80.87 |
| VMamba-T | 691 | 78.29 | 79.49 | 79.19 | 89.60 | 66.66 | [70.6, 85.4] | 78.33 (73.94–89.05) | 78.88 |

Paired tests, Hybrid-res vs baseline (pooled; patient bootstrap CI and patient sign-flip p, Holm across baselines; image-level McNemar for reference; 'secondary': VMamba-T, outside the Holm family, p unadjusted):

| Baseline | Δ acc | Patient 95% CI | Patient p | Holm p | Discordant | McNemar p |
|---|---|---|---|---|---|---|
| CNN pathway | -0.43 | [-4.73, +3.87] | 0.899 | 0.899 | 53 / 56 | 0.848 |
| EfficientNet-B0 | +9.99 | [+6.41, +14.23] | <0.001 | <0.001 | 118 / 49 | <0.001 |
| ViT-S/16 | -3.04 | [-7.99, +2.01] | 0.281 | 0.842 | 65 / 86 | 0.104 |
| Swin-T | -2.89 | [-9.15, +2.93] | 0.380 | 0.842 | 63 / 83 | 0.116 |
| Phikon linear probe | -7.38 | [-12.61, -1.89] | 0.019 | 0.078 | 57 / 108 | <0.001 |
| VMamba-T | -5.21 | [-10.26, -0.16] | 0.063 | secondary | 48 / 84 | 0.002 |

## T5. LungHist700 7-class differentiation grading, patient-level 5-fold CV (all)
Pooled out-of-fold predictions (each image predicted once by a model that never saw its patient), in %. CIs resample patients.

| Model | N | Acc | BalAcc | F1 | AUC | MCC | Acc 95% CI (patient) | Fold acc mean (range) | Acc excl. LR-selection patients | Superclass acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid-res | 691 | 46.89 | 43.40 | 43.23 | 78.22 | 37.50 | [39.1, 54.4] | 46.99 (39.13–57.14) | 46.75 | 69.46 |
| CNN pathway | 691 | 48.63 | 45.02 | 44.61 | 77.95 | 39.57 | [41.7, 55.5] | 48.43 (41.80–59.86) | 48.38 | 71.92 |
| EfficientNet-B0 | 691 | 39.80 | 37.71 | 37.24 | 76.12 | 29.73 | [33.1, 46.2] | 39.89 (35.40–44.22) | 39.17 | 65.56 |
| ViT-S/16 | 691 | 46.16 | 42.03 | 42.51 | 74.69 | 36.77 | [38.1, 53.9] | 46.08 (42.28–49.28) | 47.11 | 71.20 |
| Swin-T | 691 | 46.74 | 42.26 | 42.34 | 75.63 | 37.29 | [39.2, 53.8] | 46.44 (42.03–53.74) | 46.21 | 75.40 |
| Phikon linear probe | 691 | 50.22 | 45.89 | 45.91 | 81.39 | 41.31 | [41.0, 58.5] | 50.05 (46.34–61.22) | 51.26 | 78.15 |
| VMamba-T | 691 | 46.31 | 41.54 | 41.39 | 74.10 | 36.73 | [38.0, 54.4] | 46.27 (40.37–53.74) | 46.57 | 75.54 |

Paired tests, Hybrid-res vs baseline (pooled; patient bootstrap CI and patient sign-flip p, Holm across baselines; image-level McNemar for reference; 'secondary': VMamba-T, outside the Holm family, p unadjusted):

| Baseline | Δ acc | Patient 95% CI | Patient p | Holm p | Discordant | McNemar p |
|---|---|---|---|---|---|---|
| CNN pathway | -1.74 | [-6.21, +2.72] | 0.508 | 1.000 | 62 / 74 | 0.346 |
| EfficientNet-B0 | +7.09 | [+2.15, +12.13] | 0.008 | 0.042 | 110 / 61 | <0.001 |
| ViT-S/16 | +0.72 | [-4.68, +6.36] | 0.846 | 1.000 | 83 / 78 | 0.753 |
| Swin-T | +0.14 | [-4.87, +5.57] | 1.000 | 1.000 | 81 / 80 | 1.000 |
| Phikon linear probe | -3.33 | [-8.30, +2.44] | 0.295 | 1.000 | 77 / 100 | 0.098 |
| VMamba-T | +0.58 | [-4.74, +5.64] | 0.869 | secondary | 76 / 72 | 0.805 |

## T6. External test: LC25000-trained models on LungHist700 (3-class)
Mean (minimum–maximum) over seeds (n = seeds), in %. No retraining or adaptation.

| Model | 20× n | 20× Acc | 20× BalAcc | 20× F1 | 20× AUC | 40× n | 40× Acc | 40× BalAcc | 40× F1 | 40× AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid-res | 3 | 42.43 (35.65–48.19) | 38.56 (31.31–44.79) | 34.89 (28.62–39.49) | 57.68 (46.36–66.48) | 3 | 58.73 (56.63–61.45) | 53.01 (49.46–56.06) | 53.07 (48.35–55.85) | 72.90 (65.47–78.30) |
| Hybrid-res-xattn | 1 | 44.01 | 40.77 | 31.99 | 66.74 | 1 | 56.63 | 49.27 | 46.79 | 77.09 |
| Original hybrid | 1 | 47.08 | 42.21 | 34.59 | 60.51 | 1 | 56.63 | 47.97 | 43.76 | 65.89 |
| CNN pathway | 3 | 45.40 (43.18–46.80) | 40.66 (38.83–42.11) | 34.10 (33.29–34.84) | 62.87 (61.18–63.88) | 3 | 55.12 (52.71–56.93) | 47.70 (45.95–48.96) | 45.81 (44.72–46.39) | 73.25 (72.76–73.92) |
| EfficientNet-B0 | 3 | 39.28 (37.60–40.67) | 36.58 (34.52–37.91) | 31.95 (28.36–36.32) | 52.25 (48.82–55.54) | 3 | 54.82 (52.11–58.13) | 47.44 (44.53–49.50) | 43.90 (39.54–46.90) | 71.37 (68.03–75.15) |
| ViT-S/16 | 3 | 51.62 (48.75–56.55) | 46.77 (43.80–51.22) | 43.67 (41.01–47.77) | 67.11 (64.39–71.83) | 3 | 62.85 (60.84–65.66) | 56.08 (53.24–59.26) | 56.21 (52.02–60.12) | 78.06 (73.04–81.70) |
| Swin-T | 3 | 47.63 (43.18–50.70) | 42.72 (38.02–46.66) | 37.78 (32.67–43.85) | 61.20 (54.24–68.34) | 3 | 60.04 (59.04–61.75) | 51.45 (49.42–55.26) | 48.30 (44.70–55.46) | 69.75 (68.64–71.63) |
| Phikon linear probe | 3 | 50.05 (49.03–50.70) | 49.78 (48.75–51.77) | 46.93 (44.69–50.32) | 72.65 (70.32–74.90) | 3 | 73.39 (72.89–74.10) | 69.98 (69.87–70.16) | 71.70 (71.57–71.87) | 91.56 (90.67–92.09) |
| VMamba-T | 3 | 52.46 (51.25–53.20) | 47.17 (46.85–47.49) | 44.01 (43.38–45.26) | 66.42 (65.19–68.22) | 3 | 63.45 (62.65–64.16) | 55.74 (54.23–57.01) | 54.78 (52.15–56.80) | 75.52 (74.22–76.91) |

## T7. Sensitivity: strictly clean LC25000 test subset (seed 42)
Clean = test images whose LC25000-clean tile has no member in training (2221 of 2261).

| Model | All test | Clean subset | Δ |
|---|---|---|---|
| Hybrid-res | 98.10 | 98.11 | +0.01 |
| Hybrid-res-xattn | 99.34 | 99.37 | +0.03 |
| Original hybrid | 98.94 | 98.92 | -0.02 |
| CNN pathway | 99.16 | 99.14 | -0.02 |
| EfficientNet-B0 | 97.83 | 97.84 | +0.01 |
| ViT-S/16 | 99.29 | 99.28 | -0.01 |
| Swin-T | 98.63 | 98.60 | -0.02 |
| Phikon linear probe | 99.29 | 99.28 | -0.01 |
| VMamba-T | 98.36 | 98.33 | -0.03 |

## T8a. Saliency faithfulness on LC25000 (blur canvas, 300 test images)
Insertion − deletion AUC (higher = more faithful); 'vs random' is the paired difference from a random map on the same 300 images. Group p: sign-flip test on per-source-group mean differences (primary), Holm across all rows of the canvas; image-level Wilcoxon p for reference.

| Run | Map | Deletion | Insertion | Ins − Del | vs random | Wilcoxon p | Group p | Group Holm p |
|---|---|---|---|---|---|---|---|---|
| hybrid_res_d4_224px_s1337 | cnn | 0.737 | 0.756 | 0.019 | +0.022 | <0.001 | 0.012 | 0.158 |
| hybrid_res_d4_224px_s1337 | mamba | 0.706 | 0.779 | 0.073 | +0.076 | <0.001 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s1337 | combined | 0.720 | 0.770 | 0.051 | +0.054 | <0.001 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s2024 | cnn | 0.735 | 0.773 | 0.038 | +0.043 | <0.001 | 0.001 | 0.021 |
| hybrid_res_d4_224px_s2024 | mamba | 0.769 | 0.744 | -0.025 | -0.020 | 0.217 | 0.402 | 1.000 |
| hybrid_res_d4_224px_s2024 | combined | 0.745 | 0.766 | 0.020 | +0.025 | <0.001 | 0.060 | 0.580 |
| hybrid_res_d4_224px_s42 | cnn | 0.837 | 0.838 | 0.002 | +0.004 | 0.002 | 0.889 | 1.000 |
| hybrid_res_d4_224px_s42 | mamba | 0.812 | 0.855 | 0.043 | +0.046 | 0.486 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s42 | combined | 0.821 | 0.852 | 0.031 | +0.033 | 0.061 | 0.006 | 0.091 |
| hybrid_res_xattn_d4_224px_s42 | cnn | 0.831 | 0.858 | 0.026 | +0.031 | <0.001 | <0.001 | 0.011 |
| hybrid_res_xattn_d4_224px_s42 | mamba | 0.800 | 0.858 | 0.059 | +0.063 | <0.001 | <0.001 | 0.003 |
| hybrid_res_xattn_d4_224px_s42 | combined | 0.803 | 0.864 | 0.062 | +0.066 | <0.001 | <0.001 | 0.003 |
| grouped_d4_224px_s42 | cnn | 0.863 | 0.892 | 0.029 | +0.031 | <0.001 | <0.001 | 0.003 |
| grouped_d4_224px_s42 | mamba | 0.878 | 0.887 | 0.008 | +0.010 | 0.363 | 0.261 | 1.000 |
| grouped_d4_224px_s42 | combined | 0.868 | 0.885 | 0.017 | +0.019 | <0.001 | 0.058 | 0.580 |
| cnn_only_d4_224px_s1337 | cnn | 0.748 | 0.825 | 0.078 | +0.083 | <0.001 | <0.001 | 0.003 |
| cnn_only_d4_224px_s2024 | cnn | 0.829 | 0.876 | 0.047 | +0.050 | <0.001 | <0.001 | 0.003 |
| cnn_only_d4_224px_s42 | cnn | 0.780 | 0.839 | 0.059 | +0.063 | <0.001 | <0.001 | 0.003 |
| effnet_b0_d4_224px_s1337 | cnn | 0.829 | 0.847 | 0.018 | +0.016 | 0.104 | 0.045 | 0.499 |
| effnet_b0_d4_224px_s2024 | cnn | 0.861 | 0.868 | 0.007 | +0.017 | 0.004 | 0.147 | 1.000 |
| effnet_b0_d4_224px_s42 | cnn | 0.796 | 0.827 | 0.031 | +0.037 | 0.001 | 0.011 | 0.158 |
| vit_small_d4_224px_s1337 | vit | 0.847 | 0.867 | 0.020 | +0.024 | <0.001 | <0.001 | 0.011 |
| vit_small_d4_224px_s2024 | vit | 0.895 | 0.913 | 0.018 | +0.017 | <0.001 | 0.014 | 0.163 |
| vit_small_d4_224px_s42 | vit | 0.886 | 0.891 | 0.005 | +0.004 | 0.028 | 0.811 | 1.000 |
| swin_tiny_d4_224px_s1337 | swin | 0.862 | 0.862 | 0.001 | +0.003 | 0.017 | 0.748 | 1.000 |
| swin_tiny_d4_224px_s2024 | swin | 0.908 | 0.915 | 0.007 | +0.001 | <0.001 | 0.905 | 1.000 |
| swin_tiny_d4_224px_s42 | swin | 0.906 | 0.910 | 0.004 | +0.006 | 0.325 | 0.149 | 1.000 |

## T8a-sum. Faithfulness summary per architecture (blur canvas)
Δ vs random: mean (minimum to maximum) over seeds. Significant seeds: group Holm p < 0.05, in either direction; seeds in which the map was significantly below the random map are counted in brackets.

| Model | Map | Seeds | Δ vs random | Significant seeds |
|---|---|---|---|---|
| Hybrid-res | cnn | 3 | +0.023 (+0.004 to +0.043) | 1/3 |
| Hybrid-res | mamba | 3 | +0.034 (-0.020 to +0.076) | 2/3 |
| Hybrid-res | combined | 3 | +0.038 (+0.025 to +0.054) | 1/3 |
| Hybrid-res-xattn | cnn | 1 | +0.031 | 1/1 |
| Hybrid-res-xattn | mamba | 1 | +0.063 | 1/1 |
| Hybrid-res-xattn | combined | 1 | +0.066 | 1/1 |
| Original hybrid | cnn | 1 | +0.031 | 1/1 |
| Original hybrid | mamba | 1 | +0.010 | 0/1 |
| Original hybrid | combined | 1 | +0.019 | 0/1 |
| CNN pathway | cnn | 3 | +0.065 (+0.050 to +0.083) | 3/3 |
| EfficientNet-B0 | cnn | 3 | +0.024 (+0.016 to +0.037) | 0/3 |
| ViT-S/16 | vit | 3 | +0.015 (+0.004 to +0.024) | 1/3 |
| Swin-T | swin | 3 | +0.003 (+0.001 to +0.006) | 0/3 |

## T8d. Saliency faithfulness on LC25000 (mean-colour canvas, secondary)
Insertion − deletion AUC (higher = more faithful); 'vs random' is the paired difference from a random map on the same 300 images. Group p: sign-flip test on per-source-group mean differences (primary), Holm across all rows of the canvas; image-level Wilcoxon p for reference.

| Run | Map | Deletion | Insertion | Ins − Del | vs random | Wilcoxon p | Group p | Group Holm p |
|---|---|---|---|---|---|---|---|---|
| hybrid_res_d4_224px_s1337 | cnn | 0.712 | 0.756 | 0.044 | -0.011 | 0.835 | 0.676 | 1.000 |
| hybrid_res_d4_224px_s1337 | mamba | 0.647 | 0.779 | 0.132 | +0.076 | <0.001 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s1337 | combined | 0.669 | 0.770 | 0.102 | +0.047 | <0.001 | 0.003 | 0.023 |
| hybrid_res_d4_224px_s2024 | cnn | 0.719 | 0.773 | 0.054 | -0.047 | <0.001 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s2024 | mamba | 0.703 | 0.744 | 0.041 | -0.060 | <0.001 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s2024 | combined | 0.719 | 0.766 | 0.047 | -0.054 | <0.001 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s42 | cnn | 0.740 | 0.838 | 0.098 | -0.045 | <0.001 | <0.001 | 0.003 |
| hybrid_res_d4_224px_s42 | mamba | 0.688 | 0.855 | 0.167 | +0.023 | 0.074 | 0.217 | 1.000 |
| hybrid_res_d4_224px_s42 | combined | 0.713 | 0.852 | 0.139 | -0.005 | <0.001 | 0.195 | 1.000 |
| hybrid_res_xattn_d4_224px_s42 | cnn | 0.722 | 0.858 | 0.136 | -0.049 | <0.001 | <0.001 | 0.003 |
| hybrid_res_xattn_d4_224px_s42 | mamba | 0.629 | 0.858 | 0.229 | +0.044 | 0.001 | 0.002 | 0.020 |
| hybrid_res_xattn_d4_224px_s42 | combined | 0.671 | 0.864 | 0.194 | +0.009 | 0.017 | 0.556 | 1.000 |
| grouped_d4_224px_s42 | cnn | 0.732 | 0.892 | 0.160 | -0.009 | <0.001 | 0.303 | 1.000 |
| grouped_d4_224px_s42 | mamba | 0.763 | 0.887 | 0.124 | -0.046 | <0.001 | <0.001 | 0.003 |
| grouped_d4_224px_s42 | combined | 0.740 | 0.885 | 0.146 | -0.024 | <0.001 | 0.041 | 0.288 |
| cnn_only_d4_224px_s1337 | cnn | 0.619 | 0.825 | 0.206 | +0.050 | <0.001 | <0.001 | 0.003 |
| cnn_only_d4_224px_s2024 | cnn | 0.685 | 0.876 | 0.192 | +0.027 | 0.010 | 0.008 | 0.065 |
| cnn_only_d4_224px_s42 | cnn | 0.641 | 0.839 | 0.197 | +0.060 | <0.001 | <0.001 | 0.003 |
| effnet_b0_d4_224px_s1337 | cnn | 0.543 | 0.847 | 0.303 | +0.170 | <0.001 | <0.001 | 0.003 |
| effnet_b0_d4_224px_s2024 | cnn | 0.497 | 0.868 | 0.372 | +0.303 | <0.001 | <0.001 | 0.003 |
| effnet_b0_d4_224px_s42 | cnn | 0.515 | 0.827 | 0.311 | +0.196 | <0.001 | <0.001 | 0.003 |
| vit_small_d4_224px_s1337 | vit | 0.645 | 0.867 | 0.222 | +0.068 | <0.001 | <0.001 | 0.003 |
| vit_small_d4_224px_s2024 | vit | 0.676 | 0.913 | 0.238 | -0.002 | 0.237 | 0.433 | 1.000 |
| vit_small_d4_224px_s42 | vit | 0.643 | 0.891 | 0.248 | +0.062 | <0.001 | <0.001 | 0.003 |
| swin_tiny_d4_224px_s1337 | swin | 0.707 | 0.862 | 0.155 | +0.109 | <0.001 | <0.001 | 0.003 |
| swin_tiny_d4_224px_s2024 | swin | 0.794 | 0.915 | 0.121 | +0.073 | <0.001 | <0.001 | 0.003 |
| swin_tiny_d4_224px_s42 | swin | 0.732 | 0.910 | 0.178 | +0.148 | <0.001 | <0.001 | 0.003 |

Summary over seeds (mean-colour canvas):

| Model | Map | Seeds | Δ vs random | Significant seeds |
|---|---|---|---|---|
| Hybrid-res | cnn | 3 | -0.034 (-0.047 to -0.011) | 2/3 (2 below random) |
| Hybrid-res | mamba | 3 | +0.013 (-0.060 to +0.076) | 2/3 (1 below random) |
| Hybrid-res | combined | 3 | -0.004 (-0.054 to +0.047) | 2/3 (1 below random) |
| Hybrid-res-xattn | cnn | 1 | -0.049 | 1/1 (1 below random) |
| Hybrid-res-xattn | mamba | 1 | +0.044 | 1/1 |
| Hybrid-res-xattn | combined | 1 | +0.009 | 0/1 |
| Original hybrid | cnn | 1 | -0.009 | 0/1 |
| Original hybrid | mamba | 1 | -0.046 | 1/1 (1 below random) |
| Original hybrid | combined | 1 | -0.024 | 0/1 |
| CNN pathway | cnn | 3 | +0.046 (+0.027 to +0.060) | 2/3 |
| EfficientNet-B0 | cnn | 3 | +0.223 (+0.170 to +0.303) | 3/3 |
| ViT-S/16 | vit | 3 | +0.043 (-0.002 to +0.068) | 2/3 |
| Swin-T | swin | 3 | +0.110 (+0.073 to +0.148) | 3/3 |

## T8b. Pathway attribution (residual hybrids, full LC25000 test set)

| Run | Acc full | Acc Mamba ablated | Flipped | McNemar p | ‖δ‖/‖f_cnn‖ | Prob drop |
|---|---|---|---|---|---|---|
| hybrid_res_d4_224px_s1337 | 95.63 | 86.24 | 324 | <0.001 | 1.062 | 0.3995 |
| hybrid_res_d4_224px_s2024 | 97.73 | 85.97 | 332 | <0.001 | 1.560 | 0.4192 |
| hybrid_res_d4_224px_s42 | 98.10 | 90.40 | 228 | <0.001 | 1.782 | 0.4415 |
| hybrid_res_xattn_d4_224px_s42 | 99.34 | 85.71 | 324 | <0.001 | 1.837 | 0.4798 |

## T8c. LungHist700 explanations, patient-level folds (all)
(a) Faithfulness (blur canvas): Δ = mean over patients of the per-image difference in insertion − deletion AUC between the map and a random map; patient bootstrap 95% CI; '% patients > 0'; sign-flip p over patients, Holm across all rows.

| Model | Map | Images | Patients | Map ins−del | Random ins−del | Δ vs random [95% CI] | % patients > 0 | Holm p |
|---|---|---|---|---|---|---|---|---|
| Hybrid-res | cnn | 691 | 45 | 0.170 | -0.004 | +0.174 [+0.159, +0.191] | 100 | <0.001 |
| Hybrid-res | mamba | 691 | 45 | 0.028 | -0.004 | +0.046 [+0.026, +0.069] | 71 | <0.001 |
| Hybrid-res | combined | 691 | 45 | 0.141 | -0.004 | +0.151 [+0.132, +0.173] | 100 | <0.001 |
| CNN pathway | cnn | 691 | 45 | 0.175 | -0.006 | +0.181 [+0.162, +0.201] | 100 | <0.001 |
| EfficientNet-B0 | cnn | 691 | 45 | 0.046 | -0.003 | +0.053 [+0.023, +0.082] | 80 | 0.003 |
| ViT-S/16 | vit | 691 | 45 | -0.007 | 0.001 | -0.009 [-0.045, +0.028] | 47 | 0.644 |
| Swin-T | swin | 691 | 45 | 0.057 | 0.002 | +0.052 [+0.025, +0.081] | 78 | 0.001 |

(a2) Faithfulness, mean-colour canvas (deleted pixels replaced by the normalisation mean; insertion as for the blur canvas); same statistics, Holm across the rows of this canvas.

| Model | Map | Images | Patients | Map ins−del | Random ins−del | Δ vs random [95% CI] | % patients > 0 | Holm p |
|---|---|---|---|---|---|---|---|---|
| Hybrid-res | cnn | 691 | 45 | 0.185 | 0.036 | +0.136 [+0.110, +0.163] | 91 | <0.001 |
| Hybrid-res | mamba | 691 | 45 | 0.078 | 0.036 | +0.068 [+0.038, +0.102] | 73 | <0.001 |
| Hybrid-res | combined | 691 | 45 | 0.160 | 0.036 | +0.124 [+0.103, +0.144] | 98 | <0.001 |
| CNN pathway | cnn | 691 | 45 | 0.164 | -0.000 | +0.156 [+0.130, +0.181] | 96 | <0.001 |
| EfficientNet-B0 | cnn | 691 | 45 | 0.062 | -0.047 | +0.114 [+0.085, +0.144] | 84 | <0.001 |
| ViT-S/16 | vit | 691 | 45 | 0.128 | 0.039 | +0.083 [+0.046, +0.122] | 76 | <0.001 |
| Swin-T | swin | 691 | 45 | 0.142 | 0.010 | +0.137 [+0.087, +0.190] | 82 | <0.001 |

(b) Pathway attribution of Hybrid-res, pooled out-of-fold; accuracy change = full − Mamba term removed, patient bootstrap CI and sign-flip p.

| Task | N | Acc full | Acc Mamba ablated | Change [95% CI] | Patient p | Flipped | ‖δ‖/‖f_cnn‖ |
|---|---|---|---|---|---|---|---|
| 3c | 691 | 73.08 | 72.36 | +0.72 [-1.02, +2.27] | 0.516 | 24 (3.5%) | 0.190 |
| 7c | 691 | 46.89 | 46.45 | +0.43 [-1.30, +2.54] | 0.876 | 44 (6.4%) | 0.163 |

## T9. Computational cost (NVIDIA A10G, 224 px, fp16=True)
GFLOPs exclude the element-wise selective scan (lower bound for Mamba models). VMamba-T was measured in a second session on the same instance type and software, in which Hybrid-res took 92.25 ms at batch 1 (91.54 ms above) and Swin-T took 10.94 ms at batch 1 (10.93 ms above).

| Model | Params (M) | GFLOPs | Latency b1 (ms) | Throughput b64 (img/s) | Peak mem b64 (MB) |
|---|---|---|---|---|---|
| Hybrid-res | 9.55 | 3.00 | 91.54 | 73 | 1188 |
| Hybrid-res-xattn | 9.96 | 3.09 | 89.15 | 73 | 1191 |
| Original hybrid | 10.05 | 3.14 | 92.43 | 73 | 1218 |
| CNN pathway | 3.65 | 0.73 | 7.72 | 4154 | 404 |
| EfficientNet-B0 | 4.01 | 0.77 | 7.28 | 4166 | 440 |
| ViT-S/16 | 21.67 | 9.20 | 5.25 | 3168 | 319 |
| Swin-T | 27.52 | 8.98 | 10.93 | 1300 | 967 |
| VMamba-T (second session) | 29.48 | 9.62 | 18.21 | 949 | 1145 |

## T10. LungHist700 learning-rate sensitivity (added after the pre-submission review)
Fold-0 validation accuracy (%, three-class patient-level split) at each rate; the pre-specified choice is the better of 1e-4 and 1e-3, the sensitivity rate the best of 1e-4, 1e-3 and 3e-3 (ties keep the smaller rate). Primary results are those at the pre-specified rate.

| Model | 1e-4 | 1e-3 | 3e-3 | Pre-specified | Sensitivity |
|---|---|---|---|---|---|
| Hybrid-res | 71.5 | 75.2 | 72.3 | 1e-3 | 1e-3 |
| CNN pathway | 70.8 | 78.1 | 78.1 | 1e-3 | 1e-3 |
| EfficientNet-B0 | 51.1 | 69.3 | 76.6 | 1e-3 | 3e-3 |
| ViT-S/16 | 73.7 | 77.4 | 73.7 | 1e-3 | 1e-3 |
| Swin-T | 71.5 | 78.8 | 76.6 | 1e-3 | 1e-3 |
| VMamba-T | 76.6 | 81.8 | 78.8 | 1e-3 | 1e-3 |

LungHist700, three classes, pooled out-of-fold, models whose sensitivity rate differs from the pre-specified one, at that rate:

| Model (rate) | N | Acc | BalAcc | F1 | AUC | MCC | Acc 95% CI (patient) | Fold acc mean (range) | Acc excl. LR-selection patients |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 (3e-3) | 691 | 72.65 | 73.67 | 73.35 | 86.04 | 57.82 | [66.5, 78.6] | 72.68 (66.42–79.41) | 74.73 |

Paired tests repeated with every model at its sensitivity rate (three classes; as in T4, Holm across the same five comparisons):

| Baseline | Δ acc | Patient 95% CI | Patient p | Holm p | Discordant | McNemar p |
|---|---|---|---|---|---|---|
| CNN pathway | -0.43 | [-4.73, +3.87] | 0.899 | 1.000 | 53 / 56 | 0.848 |
| EfficientNet-B0 | +0.43 | [-3.55, +4.74] | 0.893 | 1.000 | 73 / 70 | 0.867 |
| ViT-S/16 | -3.04 | [-7.99, +2.01] | 0.281 | 1.000 | 65 / 86 | 0.104 |
| Swin-T | -2.89 | [-9.15, +2.93] | 0.380 | 1.000 | 63 / 83 | 0.116 |
| Phikon linear probe | -7.38 | [-12.61, -1.89] | 0.019 | 0.097 | 57 / 108 | <0.001 |
| VMamba-T | -5.21 | [-10.26, -0.16] | 0.063 | secondary | 48 / 84 | 0.002 |

LungHist700, seven classes, pooled out-of-fold, models whose sensitivity rate differs from the pre-specified one, at that rate:

| Model (rate) | N | Acc | BalAcc | F1 | AUC | MCC | Acc 95% CI (patient) | Fold acc mean (range) | Acc excl. LR-selection patients | Superclass acc |
|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 (3e-3) | 691 | 45.30 | 41.87 | 41.33 | 78.03 | 35.63 | [37.4, 52.7] | 45.32 (39.75–55.78) | 44.58 | 72.65 |

Paired tests repeated with every model at its sensitivity rate (seven classes; as in T5, Holm across the same five comparisons):

| Baseline | Δ acc | Patient 95% CI | Patient p | Holm p | Discordant | McNemar p |
|---|---|---|---|---|---|---|
| CNN pathway | -1.74 | [-6.21, +2.72] | 0.508 | 1.000 | 62 / 74 | 0.346 |
| EfficientNet-B0 | +1.59 | [-2.63, +6.12] | 0.539 | 1.000 | 82 / 71 | 0.419 |
| ViT-S/16 | +0.72 | [-4.68, +6.36] | 0.846 | 1.000 | 83 / 78 | 0.753 |
| Swin-T | +0.14 | [-4.87, +5.57] | 1.000 | 1.000 | 81 / 80 | 1.000 |
| Phikon linear probe | -3.33 | [-8.30, +2.44] | 0.295 | 1.000 | 77 / 100 | 0.098 |
| VMamba-T | +0.58 | [-4.74, +5.64] | 0.869 | secondary | 76 / 72 | 0.805 |

## T11. Faithfulness against a 7 × 7 random map (sensitivity; added after the pre-submission review)
Margin = map minus random map in insertion − deletion AUC, on the same images. LC25000: mean (minimum to maximum) over seeds of the image-level margin; 'significant' counts seeds with source-group Holm p < 0.05, Holm across all rows of a canvas and grid (T8). LungHist700: mean over patients [patient bootstrap 95% CI] and Holm p over the rows of a canvas and grid (T8c). The 14 × 14 columns repeat the primary analysis.

| Dataset | Canvas | Model | Map | Seeds / patients | 14 × 14: margin | 14 × 14: significant / Holm p | 7 × 7: margin | 7 × 7: significant / Holm p |
|---|---|---|---|---|---|---|---|---|
| LC25000 | blur | Hybrid-res | cnn | 3 | +0.023 | 1/3 | +0.018 (-0.000 to +0.040) | 1/3 |
| LC25000 | blur | Hybrid-res | mamba | 3 | +0.034 | 2/3 | +0.029 (-0.023 to +0.068) | 1/3 |
| LC25000 | blur | Hybrid-res | combined | 3 | +0.038 | 1/3 | +0.032 (+0.023 to +0.046) | 1/3 |
| LC25000 | blur | Hybrid-res-xattn | cnn | 1 | +0.031 | 1/1 | +0.026 (+0.026 to +0.026) | 0/1 |
| LC25000 | blur | Hybrid-res-xattn | mamba | 1 | +0.063 | 1/1 | +0.058 (+0.058 to +0.058) | 1/1 |
| LC25000 | blur | Hybrid-res-xattn | combined | 1 | +0.066 | 1/1 | +0.061 (+0.061 to +0.061) | 1/1 |
| LC25000 | blur | Original hybrid | cnn | 1 | +0.031 | 1/1 | +0.024 (+0.024 to +0.024) | 0/1 |
| LC25000 | blur | Original hybrid | mamba | 1 | +0.010 | 0/1 | +0.003 (+0.003 to +0.003) | 0/1 |
| LC25000 | blur | Original hybrid | combined | 1 | +0.019 | 0/1 | +0.012 (+0.012 to +0.012) | 0/1 |
| LC25000 | blur | CNN pathway | cnn | 3 | +0.065 | 3/3 | +0.059 (+0.044 to +0.075) | 3/3 |
| LC25000 | blur | EfficientNet-B0 | cnn | 3 | +0.024 | 0/3 | +0.021 (+0.013 to +0.035) | 0/3 |
| LC25000 | blur | ViT-S/16 | vit | 3 | +0.015 | 1/3 | +0.011 (-0.004 to +0.020) | 0/3 |
| LC25000 | blur | Swin-T | swin | 3 | +0.003 | 0/3 | +0.002 (-0.004 to +0.009) | 0/3 |
| LC25000 | mean | Hybrid-res | cnn | 3 | -0.034 | 2/3 | +0.040 (+0.033 to +0.044) | 3/3 |
| LC25000 | mean | Hybrid-res | mamba | 3 | +0.013 | 2/3 | +0.088 (+0.031 to +0.132) | 3/3 |
| LC25000 | mean | Hybrid-res | combined | 3 | -0.004 | 2/3 | +0.071 (+0.037 to +0.102) | 3/3 |
| LC25000 | mean | Hybrid-res-xattn | cnn | 1 | -0.049 | 1/1 | +0.041 (+0.041 to +0.041) | 1/1 |
| LC25000 | mean | Hybrid-res-xattn | mamba | 1 | +0.044 | 1/1 | +0.134 (+0.134 to +0.134) | 1/1 |
| LC25000 | mean | Hybrid-res-xattn | combined | 1 | +0.009 | 0/1 | +0.098 (+0.098 to +0.098) | 1/1 |
| LC25000 | mean | Original hybrid | cnn | 1 | -0.009 | 0/1 | +0.044 (+0.044 to +0.044) | 1/1 |
| LC25000 | mean | Original hybrid | mamba | 1 | -0.046 | 1/1 | +0.007 (+0.007 to +0.007) | 0/1 |
| LC25000 | mean | Original hybrid | combined | 1 | -0.024 | 0/1 | +0.029 (+0.029 to +0.029) | 1/1 |
| LC25000 | mean | CNN pathway | cnn | 3 | +0.046 | 2/3 | +0.105 (+0.095 to +0.113) | 3/3 |
| LC25000 | mean | EfficientNet-B0 | cnn | 3 | +0.223 | 3/3 | +0.335 (+0.289 to +0.419) | 3/3 |
| LC25000 | mean | ViT-S/16 | vit | 3 | +0.043 | 2/3 | +0.098 (+0.074 to +0.118) | 3/3 |
| LC25000 | mean | Swin-T | swin | 3 | +0.110 | 3/3 | +0.131 (+0.090 to +0.157) | 3/3 |
| LungHist700 | blur | Hybrid-res | cnn | 45 | +0.174 | <0.001 | +0.171 [+0.154, +0.191] | <0.001 |
| LungHist700 | blur | Hybrid-res | mamba | 45 | +0.046 | <0.001 | +0.043 [+0.019, +0.072] | 0.001 |
| LungHist700 | blur | Hybrid-res | combined | 45 | +0.151 | <0.001 | +0.148 [+0.126, +0.173] | <0.001 |
| LungHist700 | blur | CNN pathway | cnn | 45 | +0.181 | <0.001 | +0.175 [+0.155, +0.198] | <0.001 |
| LungHist700 | blur | EfficientNet-B0 | cnn | 45 | +0.053 | 0.003 | +0.061 [+0.027, +0.097] | 0.002 |
| LungHist700 | blur | ViT-S/16 | vit | 45 | -0.009 | 0.644 | -0.009 [-0.042, +0.025] | 0.611 |
| LungHist700 | blur | Swin-T | swin | 45 | +0.052 | 0.001 | +0.058 [+0.033, +0.086] | <0.001 |
| LungHist700 | mean | Hybrid-res | cnn | 45 | +0.136 | <0.001 | +0.178 [+0.157, +0.200] | <0.001 |
| LungHist700 | mean | Hybrid-res | mamba | 45 | +0.068 | <0.001 | +0.110 [+0.079, +0.143] | <0.001 |
| LungHist700 | mean | Hybrid-res | combined | 45 | +0.124 | <0.001 | +0.166 [+0.146, +0.188] | <0.001 |
| LungHist700 | mean | CNN pathway | cnn | 45 | +0.156 | <0.001 | +0.186 [+0.166, +0.207] | <0.001 |
| LungHist700 | mean | EfficientNet-B0 | cnn | 45 | +0.114 | <0.001 | +0.157 [+0.117, +0.198] | <0.001 |
| LungHist700 | mean | ViT-S/16 | vit | 45 | +0.083 | <0.001 | +0.161 [+0.120, +0.200] | <0.001 |
| LungHist700 | mean | Swin-T | swin | 45 | +0.137 | <0.001 | +0.192 [+0.158, +0.228] | <0.001 |
