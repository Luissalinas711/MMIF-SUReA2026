# Multimodal Medical Image Fusion

Undergraduate research at CSUF separating what the *representation* contributes to medical image
fusion from what the *combination rule* contributes, using five transforms crossed with three rules
on brain scans from the Harvard Whole Brain Atlas.

**Student:** Luis Rey Salinas Jr.
**Mentor:** Dr. Yoonsuk Choi
**School:** California State University, Fullerton
**Status:** Expanding on prior research from SUReA Summer 2026.

The summer version of this work, as submitted for the SUReA poster and report, is tagged:
[`surea-2026`](../../releases/tag/surea-2026).
Everything on `main` is the expanded study.

## Overview

No single medical scanner captures everything a clinician needs. MRI resolves soft tissue, CT resolves bone, and PET and SPECT show metabolic or perfusion activity with little anatomy of their own.
Multimodal medical image fusion combines two registered scans into a single image that keeps the useful detail of both.

Every fusion method here is the same three-step move in a different representation: transform, combine, invert. 
The summer study compared four methods, but each method arrived locked to one combination rule, so a difference between two methods intertwined the effect of the representation with the effect of the rule. 
This version separates them. Any of the five transforms pairs with any of the three rules, where every combination is measured identically.
The question becomes which factor actually drives performance.

## Transforms and rules

Five transforms, from no transform at all to a learned one:

| Transform | Basis |
|---|---|
| `averaging` | none, rules act on raw pixels |
| `laplacian` | Laplacian pyramid, four levels |
| `dwt` | Daubechies-2 wavelet, two levels |
| `swt` | stationary wavelet, same wavelet and depth, no decimation |
| `densefuse` | pretrained CNN autoencoder, inference only |

Three rules, which are the same weighted combination with different weights:

| Rule | Weights |
|---|---|
| `average` | fixed and equal, never commits to a source |
| `l1norm` | proportional to local activity, lightly commits |
| `maxabs` | zero or one, completely commits |

The rule therefore is singularly focused on controlling how strongly the fusion commits to one source at each coefficient location.
In every transform/rule combination the rule applies to the detail bands only and the approximation band is always averaged, which keeps the rule meaning the same thing throughout.

Fifteen combinations are formed, but only twelve distinct results arise.  
This is because averaging coefficients in any linear basis and inverting is the same as averaging the images, so `averaging`, `laplacian`, `dwt` and `swt`under the `average` rule produce identical output. 
This is verified on the real results at 8.9e-16 and used as a correctness check on the whole pipeline.

## Dataset

Harvard Whole Brain Atlas, [AANLIB](https://www.med.harvard.edu/AANLIB/home.html).

The atlas was scanned for every case holding both an MRI and a partner modality at the same slice index, rather than picking cases by hand.
Three slices per case are sampled at even intervals across the middle of each case, since neighboring slices are near duplicates. 

| Pairing | Pairs | Cases |
|---|---|---|
| MRI-CT | 30 | 10 |
| MRI-SPECT | 65 | 22 |
| MRI-PET | 9 | 3 |
| **Total** | **104** | **33 patients** |

MRI-PET is limited to the three cases the atlas contains, so it is reported descriptively rather than tested.

Source images and fused outputs live in Google Drive and are not committed, since the atlas has its own licensing. The results file is committed.

## Metrics

Seven measures: entropy, mutual information, standard deviation, spatial frequency, SSIM (to each source) and edge preservation.

Edge preservation is the full Xydeas and Petrović Q^AB/F measure with the published constants, including the weight exponent L = 1.5. 
Two implementation notes are in `src/metrics.py`: 
the paper prints both sigmoid kappas as positive, which inverts the response given where kappa sits in their equations, 
and the orientation difference is folded so that edges pi apart count as matching, since a boundary running dark to light in the MRI often runs light to dark in the partner.

## Repository layout

```
src/
  colab_setup.py        Colab startup and finish snippets
  utils.py              load, save, display
  coeffs.py             the Decomposition container, the base band policy, fuse()
  rules.py              average, l1norm, maxabs
  metrics.py            the seven metrics
  pipeline.py           fuse_all_pairs, run_all_combinations, check_averaging_column
  transforms/           one script per transform, each with forward() and inverse()
notebooks/
  01_download_images.py     build the dataset
  01b_verify_dataset.py     check it before committing 
  02_run_fusion_grid.py     all 15 combinations, 1560 fused images
  03_run_evaluation.py      score everything into one long CSV
results/
  metrics_long.csv      one row per measurement, 10920 rows
```


## How to run

Everything is written for Google Colab.

1. Run the startup cell from `src/colab_setup.py`. It mounts Drive, clones this repo, and puts `src/` on the path.
2. Run the notebooks in order. `01` builds the dataset, `01b` verifies it, `02` runs the grid, `03` writes the results file.

`02` and `03` both expect `DATA`, `DRIVE_ROOT` and `REPO_PATH` from the startup cell. DO NOT forget the Startup cell.

Requirements: Python 3, NumPy, SciPy, PyWavelets, scikit-image, Pillow. 
PyTorch is needed only for the DenseFuse row; the other four transforms run without it.

## Results so far

Statistical testing is still to come, so these are means rather than tested differences.

**The rule matters more than expected.** Within every transform the ordering is `maxabs` above `l1norm` above `average`, with no exceptions, and the spread is large.
The Laplacian pyramid moves from 0.42 to 0.74 on edge preservation purely by changing the rule, which is wider than most gaps between transforms.

**Shift invariance shows up.** The SWT beats the DWT on both selection rules in all three pairings, which is the decimation cost appearing in the results.

**The detail leader is the Laplacian pyramid**, which tops edge preservation in every pairing under both selection rules.

**A metric behaves oddly on asymmetric pairs.** On MRI-SPECT and MRI-PET, an unfused MRI scores higher on edge preservation than any of the fifteen configurations, while on MRI-CT it places near the bottom. 
The MRI holds about 84 percent of the total edge weight on the functional pairings and only 41 percent on MRI-CT. 
The measure rewards edges arriving intact without requiring that they arrive from both sources.

## Notes on DenseFuse

DenseFuse (Li and Wu, 2019) is an autoencoder trained to reconstruct ordinary images. 
No ground truth fused image exists, so it never trains on fusing anything. 
A fusion step is inserted between the encoder and decoder at test time instead. 
The authors' pretrained weights are used, so there is no training here.

`src/transforms/densefuse.py` calls the encoder and decoder directly and skips the network's own fusion step, because the fusion is one of the three rules now. 
That is what turns DenseFuse from two fixed settings into a full row of the grid. The authors' repository,
[`hli1221/densefuse-pytorch`](https://github.com/hli1221/densefuse-pytorch), ships both the network
and the weights and is cloned at runtime rather than committed. Point `DENSEFUSE_REPO` at it.

## References

- C. S. Xydeas and V. Petrović, "Objective image fusion performance measure," *Electronics Letters*,
  vol. 36, no. 4, pp. 308-309, 2000.
- V. Petrović and C. S. Xydeas, "Gradient-based multiresolution image fusion," *IEEE Transactions on
  Image Processing*, vol. 13, no. 2, pp. 228-237, 2004.
- S. Li, B. Yang, and J. Hu, "Performance comparison of different multi-resolution transforms for
  image fusion," *Information Fusion*, vol. 12, no. 2, pp. 74-84, 2011.
- Z. Zhang and R. S. Blum, "A categorization of multiscale-decomposition-based image fusion schemes
  with a performance study for a digital camera application," *Proceedings of the IEEE*, vol. 87,
  no. 8, pp. 1315-1326, 1999.
- H. Li and X.-J. Wu, "DenseFuse: A fusion approach to infrared and visible images," *IEEE
  Transactions on Image Processing*, vol. 28, no. 5, pp. 2614-2623, 2019.
- P. J. Burt and E. H. Adelson, "The Laplacian pyramid as a compact image code," *IEEE Transactions
  on Communications*, vol. 31, no. 4, pp. 532-540, 1983.
- S. Mallat, "A theory for multiresolution signal decomposition: the wavelet representation," *IEEE
  Transactions on Pattern Analysis and Machine Intelligence*, vol. 11, no. 7, pp. 674-693, 1989.
- Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error
  visibility to structural similarity," *IEEE Transactions on Image Processing*, vol. 13, no. 4,
  pp. 600-612, 2004.
- K. A. Johnson and J. A. Becker, *The Whole Brain Atlas (AANLIB)*, Harvard Medical School.
