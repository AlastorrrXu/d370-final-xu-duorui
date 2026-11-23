# Extension: what is left without color? (generated from the saved results)

locked 2026-11-17T00:04:27+00:00, test scored 2026-11-17T00:05:06+00:00, 20 models, seeds [42, 1, 2, 3, 4]

## Seed 42 (n = 598 test images)

| model | trained on | color test | gray test |
|---|---|---|---|
| frozen (head only) | color | 97.66% (584/598) | 97.16% (581/598) |
| frozen (head only) | gray | 96.49% (577/598) | 97.99% (586/598) |
| fine-tune (layer4 + head) | color | 98.16% (587/598) | 97.32% (582/598) |
| fine-tune (layer4 + head) | gray | 97.83% (585/598) | 96.99% (580/598) |

| comparison | difference (pp) | 95% CI | McNemar p | macro-F1 difference [95% CI] |
|---|---|---|---|---|
| P1 frozen: gray - color | +0.33 | [-0.84, +1.51] | 0.7744 | +0.0033 [-0.0083, +0.0146] |
| P2 fine-tune: gray - color | -1.17 | [-2.17, -0.33] | 0.0391 | -0.0119 [-0.0225, -0.0028] |
| S1 gray: fine-tune - frozen | -1.00 | [-2.51, +0.33] | 0.2379 | -0.0103 [-0.0253, +0.0036] |
| D1 frozen color model: gray test - color test | -0.50 | [-1.51, +0.50] | 0.5078 |  |
| D2 fine-tune color model: gray test - color test | -0.84 | [-1.84, +0.00] | 0.1797 |  |
| D3 frozen, gray test: trained on gray - on color | +0.84 | [-0.33, +2.01] | 0.2668 |  |
| D4 fine-tune, gray test: trained on gray - on color | -0.33 | [-1.34, +0.50] | 0.7266 |  |
| S2 (fine-tune - frozen in gray) - (fine-tune - frozen in color) | -1.51 | [-3.01, +0.00] |  |  |

- frozen (head only): errors 14 in color, 12 in gray
- fine-tune (layer4 + head): errors 11 in color, 18 in gray

## All seeds [42, 1, 2, 3, 4]

|  | color-trained frozen (%) | color-trained finetune (%) | gray-trained frozen (%) | gray-trained finetune (%) | frozen: gray - color (pp) | fine-tune: gray - color (pp) | fine-tune - frozen, color (pp) | fine-tune - frozen, gray (pp) | difference in differences (pp) |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 97.66 | 98.16 | 97.99 | 96.99 | 0.33 | -1.17 | 0.50 | -1.00 | -1.51 |
| 1 | 96.66 | 97.66 | 97.99 | 95.82 | 1.34 | -1.84 | 1.00 | -2.17 | -3.18 |
| 2 | 96.49 | 95.82 | 97.99 | 96.49 | 1.51 | 0.67 | -0.67 | -1.51 | -0.84 |
| 3 | 97.16 | 96.99 | 97.49 | 97.99 | 0.33 | 1.00 | -0.17 | 0.50 | 0.67 |
| 4 | 97.32 | 97.49 | 97.66 | 98.16 | 0.33 | 0.67 | 0.17 | 0.50 | 0.33 |
| mean | 97.06 | 97.22 | 97.83 | 97.09 | 0.77 | -0.13 | 0.17 | -0.74 | -0.90 |
| SD | 0.48 | 0.89 | 0.24 | 0.99 | 0.60 | 1.28 | 0.64 | 1.20 | 1.54 |

95% intervals for the means (bootstrap over seeds and test images):

- frozen: gray - color (pp): mean +0.77, [-0.37, +1.97]
- fine-tune: gray - color (pp): mean -0.13, [-1.47, +1.14]
- fine-tune - frozen, color (pp): mean +0.17, [-0.94, +1.27]
- fine-tune - frozen, gray (pp): mean -0.74, [-2.21, +0.64]
- difference in differences (pp): mean -0.90, [-2.71, +0.74]

## Recall by breed (%), mean over seeds

| breed | frozen: color | frozen: gray | finetune: color | finetune: gray | frozen: change | finetune: change |
|---|---|---|---|---|---|---|
| Abyssinian | 90.6 | 93.1 | 91.2 | 92.2 | 2.4 | 1.0 |
| Bengal | 95.8 | 98.6 | 96.6 | 96.6 | 2.8 | 0.0 |
| Persian | 96.0 | 97.2 | 96.4 | 95.2 | 1.2 | -1.2 |
| Beagle | 100.0 | 100.0 | 99.8 | 99.8 | 0.0 | 0.0 |
| Pug | 99.8 | 98.0 | 99.6 | 98.8 | -1.8 | -0.8 |
| Samoyed | 100.0 | 100.0 | 99.6 | 99.8 | 0.0 | 0.2 |
