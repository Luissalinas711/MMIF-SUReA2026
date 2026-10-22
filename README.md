# Multimodal Medical Image Fusion

Undergraduate research at CSUF pulling apart two things that usually travel together in
fusion work: the *representation* an image is converted into, and the *combination rule*
that merges the two converted images. Five of the first crossed with three of the second,
run on brain scans from Harvard's AANLIB archive.

**Student:** Luis Rey Salinas Jr.
**Mentor:** Dr. Yoonsuk Choi
**School:** California State University, Fullerton
**Status:** Expanded study complete. Grid, statistics, shift experiment and figures are all in.

The summer version of this work, as submitted for the SUReA poster and report, is tagged:
[`surea-2026`](../../releases/tag/surea-2026).
Everything on `main` is the expanded study.

## Overview

No scanner shows everything at once. 
MRI is applied to read soft tissue, CT for bone, and PET and SPECT report metabolism or blood flow while showing almost no anatomy around it.
Fusion is the attempt to put a pair of aligned scans into a single picture that still carries what each one was good for.

Every method here is the same three steps wearing a different costume: convert, merge, convert back.
The summer study lined up four published methods, but each one came welded to its own merging (fusion) step, 
so any gap between two of them tangled the representation together with the fusion rule.
This expanded work splits them. Every transform can be run against every rule, and all fifteen pairings are scored the same way, 
which makes it possible to ask which half of the design isactually doing the work.

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

So the rules form a ladder, and what they climb is how willing the merge is to let one scan win a given coefficient outright. 
Whichever rule is in play, it is handed the detail bands only; the approximation band gets a flat mean in every cell of the grid. 
Pinning that down stops the rule from quietly meaning something different under each transform.

Fifteen combinations are formed, but only twelve distinct results arise.  
This is because averaging coefficients in any linear basis and inverting is the same as averaging the images, 
so `averaging`, `laplacian`, `dwt` and `swt` under the `average` rule produce identical output. 
This is verified on the real results at 8.9e-16 and used as a correctness check on the whole pipeline.

## Dataset

Harvard Whole Brain Atlas, [AANLIB](https://www.med.harvard.edu/AANLIB/home.html).

Rather than hand-pick cases, the atlas was swept for any case carrying both an MRI and a second modality at a matching slice index.
Three slices come out of each case, spaced across its usable middle, because adjacent slices are close to copies of one another.

| Pairing | Pairs | Cases |
|---|---|---|
| MRI-CT | 30 | 10 |
| MRI-SPECT | 65 | 22 |
| MRI-PET | 9 | 3 |
| **Total** | **104** | **33 patients** |

Only three MRI-PET cases exist in the atlas, which is too few for the signed-rank test to return anything significant, so that pairing gets described and never tested.

Source images and fused outputs live in Google Drive and are not committed, since the atlas has its own licensing. 
Every results file is committed.

## Metrics

Seven measures: entropy, mutual information, standard deviation, spatial frequency, SSIM (to each source) and edge preservation.

Edge preservation is the full Xydeas and Petrović Q^AB/F measure with the published constants, including the weight exponent L = 1.5. 
Two implementation notes are in `src/metrics.py`: 
the paper prints both sigmoid kappas as positive, which inverts the response given where kappa sits in their equations, 
and orientations are compared modulo pi, so an edge and its reversal are treated as the same edge.
That second one matters because tissue boundaries routinely flip contrast intensity between the two modalities.

## Repository layout

```
src/
  colab_setup.py        Colab startup and finish snippets
  utils.py              load, save, display
  coeffs.py             the Decomposition container, the base band policy, fuse()
  rules.py              average, l1norm, maxabs
  metrics.py            the seven metrics
  evaluation.py         scores fused images against both sources into one long CSV
  pipeline.py           fuse_all_pairs, run_all_combinations, check_averaging_column
  transforms/           one script per transform, each with forward() and inverse()
notebooks/
  01_download_images.py     build the dataset
  01b_verify_dataset        check it before committing
  02_run_fusion_grid        all 15 combinations, 1560 fused images
  03_run_evaluation         score everything into one long CSV
  04_run_statistics.py      Friedman, Wilcoxon, Holm, effect sizes
  05_run_misalignment.py    the shift experiment, 2080 fusions
  06_rule_vs_transform.py   spreads, dominance verdicts, orderings, the post hoc set
  07_paper_figures.py       the three figures, read back out of the CSVs
results/
  metrics_long.csv                one row per measurement, 12,376 rows
  metrics_misalignment.csv        the same for the shift experiment, 16,016 rows
  statistics_summary.csv          mean and std per configuration, by pairing and measure
  statistics_tests.csv            14 Friedman, 98 planned and 140 post hoc Wilcoxon rows
  cell_means.csv                  the 15 cells per measure, pooled over pairings
  cell_means_by_pairing.csv       the same split by pairing
  factor_verdicts.csv             both spread ranges and the dominant factor per measure
  factor_verdicts_by_pairing.csv  the same within each pairing
  transform_orderings.csv         whether the transform order survives a change of rule
  rule_orderings.csv              which rule leads each measure, and how often
  rule_reversals.csv              the measures where the better rule flips by transform
  post_hoc_summary.csv            averaging against both selection rules, per measure
  post_hoc_exceptions.csv         the 12 comparisons that run the other way
  unfused_references.csv          unfused MRI against the best fused result per pairing
```


## How to run

Everything is written for Google Colab.

1. Run the startup cell from `src/colab_setup.py`. It mounts Drive, clones this repo, and puts `src/` on the path.
2. Work through the notebooks in numerical order. `01` assembles the dataset and `01b` audits it,
   `02` fills the grid, `03` measures it, `04` tests it, `05` runs the shift experiment,
   `06` derives the spreads and orderings, and `07` plots.
3. Run the finish cell from `src/colab_setup.py` to push the CSVs back here.

Every notebook expects `DATA`, `DRIVE_ROOT` and `REPO_PATH` from the startup cell. DO NOT forget the Startup cell.

Note that `07` is the one script that reaches past the CSVs: its third figure reads fused PNGs back out of Drive, 
so `02` has to have run in the same Drive for it to work. 
The first two figures need nothing but the committed results.

Requirements: Python 3, NumPy, SciPy, PyWavelets, scikit-image, Pillow. 
PyTorch is needed only for the DenseFuse row; the other four transforms run without it.

## Results

Patients, not slices, are the unit. Slices collapse to a case mean first, so one patient contributes a single number to any given cell. 
Friedman screens each measure, then signed-rank tests run the seven planned contrasts, Holm-corrected inside their family.

**There is no single answer to which half matters more.**
Two of the seven scores, edge preservation and spatial frequency, go to the rule outright. 
This means its narrowest spread still clears the widest the transform manages.
The other five land ambiguous, both ranges overlapping, so the honest answer there is that it depends on where in the grid you are standing.

**The rules rank the same way everywhere; how far apart they sit does not.** 
Pooled over patients, every transform puts `maxabs` over `l1norm` over `average` on edge preservation.
Per patient that also appies in 163 of 175 cells, with the DWT on MRI-SPECT accounting for most of the misses. 
The distance involved is another matter: the Laplacian pyramid travels from 0.42 to 0.74 on that measure without the transform changing at all, 
which is a longer trip than most transform-to-transform gaps, while on standard deviation the same swap moves some transforms over ten times further than others. 
On three measures the rules even trade places depending on which transform they are sitting under.

**The SWT wins, but not for the reason we expected.**
Drop decimation and keep everything else identical and the SWT takes all sixteen contrasts on measures that look back at the sources,
while losing eight of twelve on measures that only look at the output. 
The shift experiment was built to pin that advantage on decimation directly, and it came back the wrong way round:
the SWT stays ahead at every offset, but the two converge as the partner slides out of register instead of separating.
Whatever the SWT is buying, it already owns it at perfect registration. 
Aliasing fits better, since the lead only shows up under the two nonlinear rules, though this experiment cannot tell aliasing apart from shift variance.

**The edge metric does something strange on lopsided pairs.** 
Feed the edge metric an untouched MRI from either functional pairing and it scores that above every one of the twelve configurations.
On MRI-CT the same untouched MRI sinks near the bottom. 
The split tracks edge weight: the MRI carries roughly 84 percent of it against SPECT and PET, and only 41 percent against CT. 
The measure is rewarding edges that survive intact and asking nothing about whether both inputs contributed any.

## Notes on DenseFuse

DenseFuse (Li and Wu, 2019) is an autoencoder trained to reconstruct ordinary images. 
No ground truth fused image exists, so it never trains on fusing anything. 
A fusion step is inserted between the encoder and decoder at test time instead. 
The authors' pretrained weights are used, so there is no training here.

`src/transforms/densefuse.py` calls the encoder and decoder directly and skips the network's own fusion step, because the fusion is one of the three rules now. 
That is what turns DenseFuse from two fixed settings into a full row of the grid. 
The authors' repository:
[`hli1221/densefuse-pytorch`](https://github.com/hli1221/densefuse-pytorch)  
This ships both the network and the weights and is cloned at runtime rather than committed, since it carries its own license. 
Every DenseFuse result in this work comes from commit
`4394b63e9295db1c6b7a5c3664551c90f0605f2b` 
and the weights file `models/densefuse_gray.model` comes from:  
    git clone https://github.com/hli1221/densefuse-pytorch.git  
    git -C densefuse-pytorch checkout 4394b63e9295db1c6b7a5c3664551c90f0605f2b  

Point `DENSEFUSE_REPO` at the absolute path of that clone.  
  
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
