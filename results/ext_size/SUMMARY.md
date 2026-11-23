# Part 2b: fewer photos, and changing only the color (generated from the saved results)

locked 2026-11-18T02:14:57+00:00; sizes [1, 5, 20] plus 80 (Part 2a); seeds [42, 1, 2, 3, 4]; bootstrap 10000 resamples

## Literal stopping rule (Part 1)

| protocol | epochs | selected epoch, frozen | selected epoch, fine-tune | frozen correct | fine-tune correct | macro-F1, frozen | macro-F1, fine-tune | fine-tune minus frozen (pp) | 95% CI | exact McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|
| locked protocol (plateau checks from epoch 100) | 100 | 20 | 1 | 584/598 | 587/598 | 0.98 | 0.98 | 0.50 | [-0.67, +1.67] | 0.61 |
| the assignment's literal rule (floor of 5 epochs) | 5 | 4 | 1 | 584/598 | 587/598 | 0.98 | 0.98 | 0.50 | [-0.84, +1.67] | 0.61 |

## Validation, mean correct of 120

| ('photos_per_breed', '') | ('color', 'finetune') | ('color', 'frozen') | ('gray', 'finetune') | ('gray', 'frozen') |
|---|---|---|---|---|
| 1 | 93.00 | 101.20 | 90.60 | 98.40 |
| 5 | 109.20 | 112.00 | 106.40 | 111.20 |
| 20 | 116.80 | 117.00 | 112.60 | 115.60 |

## Brightness check of the color changes

| change | pixels in the region (%) | region pixels whose color changed (%) | region pixels whose brightness moved by more than 1 of 255 (%) |
|---|---|---|---|
| pet, hue +180° | 60.96 | 98.44 | 1.58 |
| background, hue +180° | 39.04 | 96.21 | 3.84 |
| whole photo, hue +180° | 100.00 | 97.57 | 2.47 |
| whole photo, hue +90° | 100.00 | 97.57 | 1.54 |
| whole photo, hue −90° | 100.00 | 97.57 | 1.18 |
| whole photo, hue +30° | 100.00 | 96.27 | 1.62 |
| whole photo, hue −30° | 100.00 | 96.25 | 1.38 |
| whole photo, saturation 50% | 100.00 | 96.28 | 0.00 |
| whole photo, saturation 0% | 100.00 | 97.57 | 0.00 |

## Control: black-and-white models

| seed | model | changed | largest_p_change |
|---|---|---|---|
| 42 | frozen | 0 | 0.06 |
| 42 | finetune | 0 | 0.06 |
| 1 | frozen | 0 | 0.05 |
| 1 | finetune | 0 | 0.15 |
| 2 | frozen | 0 | 0.06 |
| 2 | finetune | 0 | 0.10 |
| 3 | frozen | 0 | 0.04 |
| 3 | finetune | 0 | 0.04 |
| 4 | frozen | 1 | 0.04 |
| 4 | finetune | 2 | 0.04 |

## Need, by training size

| model | photos per breed | color on color (%) | B&W on B&W (%) | B&W minus color (pp) | CI low | CI high | log loss, B&W minus color | log loss CI | B&W-trained on color photos (%) | color-trained on B&W photos (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen (head only) | 1 | 84.45 | 83.65 | -0.80 | -4.41 | 2.61 | -0.03 | [-0.114, +0.033] | 82.81 | 83.78 |
| frozen (head only) | 5 | 93.78 | 94.68 | 0.90 | -1.07 | 3.14 | 0.01 | [-0.060, +0.084] | 94.08 | 94.52 |
| frozen (head only) | 20 | 96.19 | 97.02 | 0.84 | -0.33 | 2.07 | -0.01 | [-0.027, +0.013] | 96.02 | 96.72 |
| frozen (head only) | 80 | 97.06 | 97.83 | 0.77 | -0.37 | 1.97 | -0.02 | [-0.035, +0.000] | 96.32 | 98.06 |
| fine-tune (layer4 + head) | 1 | 77.73 | 74.78 | -2.94 | -7.12 | 0.84 | 0.06 | [+0.032, +0.098] | 73.14 | 73.81 |
| fine-tune (layer4 + head) | 5 | 91.61 | 91.04 | -0.57 | -1.97 | 0.94 | 0.07 | [+0.000, +0.166] | 88.86 | 90.64 |
| fine-tune (layer4 + head) | 20 | 95.18 | 95.15 | -0.03 | -1.34 | 1.27 | -0.00 | [-0.031, +0.022] | 94.82 | 93.71 |
| fine-tune (layer4 + head) | 80 | 97.22 | 97.09 | -0.13 | -1.51 | 1.14 | -0.00 | [-0.047, +0.048] | 96.86 | 96.19 |

## Fine-tuned minus frozen, by training size (not in the lock)

| trained and tested on | photos per breed | frozen (%) | fine-tuned (%) | fine-tuned minus frozen (pp) | CI low | CI high |
|---|---|---|---|---|---|---|
| color | 1 | 84.45 | 77.73 | -6.72 | -10.60 | -2.94 |
| color | 5 | 93.78 | 91.61 | -2.17 | -4.08 | -0.17 |
| color | 20 | 96.19 | 95.18 | -1.00 | -2.54 | 0.50 |
| color | 80 | 97.06 | 97.22 | 0.17 | -0.94 | 1.27 |
| gray | 1 | 83.65 | 74.78 | -8.86 | -13.21 | -4.78 |
| gray | 5 | 94.68 | 91.04 | -3.65 | -5.28 | -2.04 |
| gray | 20 | 97.02 | 95.15 | -1.87 | -3.61 | -0.27 |
| gray | 80 | 97.83 | 97.09 | -0.74 | -2.21 | 0.64 |

## Use, by training size

| model | photos per breed | original (%) | pet, hue +180° (pp) | pet, hue +180°: CI | background, hue +180° (pp) | background, hue +180°: CI | whole photo, hue +180° (pp) | whole photo, hue +180°: CI | whole photo, hue +90° (pp) | whole photo, hue −90° (pp) | whole photo, hue +30° (pp) | whole photo, hue −30° (pp) | whole photo, saturation 50% (pp) | whole photo, saturation 0% (pp) | whole photo, saturation 0%: CI | whole photo, hue ±30°, mean of both (pp) | whole photo, hue ±30°: CI | pet minus background (pp) | pet minus background: CI | wrong minus missing color (pp) | wrong minus missing color: CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen (head only) | 1 | 84.45 | -7.53 | [-9.63, -5.55] | 0.03 | [-1.24, +1.30] | -7.36 | [-9.77, -4.98] | -6.12 | -3.48 | -0.07 | -1.67 | 0.37 | -0.67 | [-2.88, +1.84] | -0.87 | [-2.16, +0.42] | -7.56 | [-9.60, -5.52] | -6.69 | [-10.43, -3.44] |
| frozen (head only) | 5 | 93.78 | -1.84 | [-3.51, -0.07] | -0.03 | [-0.87, +0.77] | -0.67 | [-2.71, +1.71] | -2.24 | -0.77 | -0.64 | 0.00 | 0.74 | 0.74 | [-0.74, +2.34] | -0.32 | [-1.30, +0.70] | -1.81 | [-3.51, +0.10] | -1.40 | [-2.84, -0.03] |
| frozen (head only) | 20 | 96.19 | -0.77 | [-1.97, +0.37] | 0.03 | [-0.70, +0.80] | -0.30 | [-1.71, +1.17] | -1.97 | -0.47 | -0.20 | -0.64 | 0.37 | 0.54 | [-0.64, +1.81] | -0.42 | [-1.15, +0.32] | -0.80 | [-1.97, +0.30] | -0.84 | [-1.91, +0.23] |
| frozen (head only) | 80 | 97.06 | -0.84 | [-2.41, +0.67] | 0.37 | [-0.40, +1.24] | -0.30 | [-1.91, +1.17] | -1.44 | -0.74 | -0.20 | -0.77 | 0.23 | 1.00 | [-0.17, +2.24] | -0.48 | [-1.34, +0.37] | -1.20 | [-2.58, +0.13] | -1.30 | [-2.48, -0.23] |
| fine-tune (layer4 + head) | 1 | 77.73 | -14.78 | [-20.67, -9.10] | -0.90 | [-2.58, +0.67] | -13.41 | [-18.29, -8.90] | -10.00 | -10.84 | -3.28 | -5.99 | -1.14 | -3.91 | [-6.79, -1.04] | -4.63 | [-7.14, -2.46] | -13.88 | [-19.93, -8.43] | -9.50 | [-14.05, -4.88] |
| fine-tune (layer4 + head) | 5 | 91.61 | -5.22 | [-7.36, -3.21] | -0.10 | [-0.94, +0.77] | -3.95 | [-6.05, -2.01] | -4.38 | -3.61 | -0.67 | -1.37 | 0.23 | -0.97 | [-2.64, +0.50] | -1.02 | [-1.87, -0.17] | -5.12 | [-7.29, -3.18] | -2.98 | [-4.98, -1.40] |
| fine-tune (layer4 + head) | 20 | 95.18 | -3.28 | [-4.75, -1.97] | -0.37 | [-1.20, +0.43] | -3.11 | [-4.48, -1.84] | -3.08 | -3.61 | -0.30 | -0.50 | -0.33 | -1.44 | [-3.04, +0.17] | -0.40 | [-1.12, +0.30] | -2.91 | [-4.38, -1.57] | -1.67 | [-3.18, -0.30] |
| fine-tune (layer4 + head) | 80 | 97.22 | -2.07 | [-3.34, -0.90] | -0.20 | [-0.84, +0.40] | -1.87 | [-3.08, -0.74] | -2.51 | -2.47 | -0.40 | -0.50 | -0.47 | -1.04 | [-2.24, +0.10] | -0.45 | [-1.19, +0.27] | -1.87 | [-3.28, -0.60] | -0.84 | [-1.81, +0.13] |

## Use by breed: pet's colors rotated by 180° (pp)

| photos per breed | model | Abyssinian | Bengal | Persian | Beagle | Pug | Samoyed |
|---|---|---|---|---|---|---|---|
| 80 | frozen (head only) | 0.82 | -5.20 | 0.80 | 0.00 | -1.00 | -0.40 |
| 80 | fine-tune (layer4 + head) | -3.88 | -3.20 | -0.80 | -3.80 | -0.60 | -0.20 |
| 1 | frozen (head only) | -8.57 | -9.80 | -16.80 | -7.20 | -2.40 | -0.40 |
| 1 | fine-tune (layer4 + head) | -11.02 | -19.40 | -18.60 | -35.80 | -3.60 | -0.20 |

## Where the recolored photos go (80 photos)

| true breed | called after the change | count |
|---|---|---|
| Bengal | Abyssinian | 39 |
| Abyssinian | Bengal | 33 |
| Abyssinian | Pug | 15 |
| Beagle | Pug | 9 |
| Beagle | Samoyed | 9 |
| Bengal | Persian | 8 |
| Pug | Abyssinian | 7 |
| Persian | Samoyed | 5 |
| Abyssinian | Persian | 3 |
| Persian | Bengal | 2 |
| Persian | Abyssinian | 2 |
| Samoyed | Abyssinian | 2 |
| Bengal | Pug | 2 |
| Pug | Beagle | 2 |
| Beagle | Abyssinian | 2 |
| Bengal | Samoyed | 1 |
| Abyssinian | Beagle | 1 |
| Samoyed | Beagle | 1 |

