# Multimodal Medical Image Fusion  

Undergraduate research at CSUF pulling apart two things that usually travel together in fusion work:  
the representation an image is converted into, and the combination rule that merges the two converted images.  
Five transforms crossed with three combination rules, run on brain scans from Harvard's AANLIB archive.  

**Student:** Luis Rey Salinas Jr.  
**Mentor:** Dr. Yoonsuk Choi  
**School:** California State University, Fullerton  
**Status:** Expanded study complete. Grid, statistics, shift experiment and figures are all in.  
The revision for the conference paper added a formal interaction test and a check of the edge metric.  

The summer version of this work, as submitted for the SUReA poster and report, is tagged:  
[`surea-2026`](../../releases/tag/surea-2026).  
The code as it stood when the conference paper was submitted is tagged:  
[`ccwc-submission`](../../releases/tag/ccwc-submission).  
The code and results behind the camera-ready paper are tagged:  
[`ccwc-camera-ready`](../../releases/tag/ccwc-camera-ready).  
Everything on `main` is the expanded study.  

## Overview

No scanner shows everything at once. 
MRI is applied to read soft tissue, CT for bone, and PET and SPECT report metabolism or blood flow while showing almost no anatomy around it.
Fusion is the attempt to put a pair of aligned scans into a single picture that still carries what each one was good for.

Every method here is the same three steps wearing a different costume: convert, merge, convert back.
The summer study lined up four published methods, but each one came welded to its own merging (fusion) step, 
so any gap between two of them tangled the representation together with the fusion rule.
This expanded work splits them. Every transform can be run against every rule, and all fifteen pairings are scored the same way, 
which makes it possible to ask which half of the design is actually doing the work.

## Transforms and rules

Five transforms, from no transform at all to a learned one:

| Transform | Basis |
|---|---|
| `averaging` | none, rules act on raw pixels (the paper calls this the identity transform) |
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
This is verified on all 104 real pairs, where the largest difference is 9.99e-16 (floating point rounding),
and used as a correctness check on the whole pipeline.

## Settings

Every setting was fixed before the grid was run and none was tuned on these images.

| Setting | Value | Where it lives |
|---|---|---|
| Image size | 256 x 256, 8-bit grayscale scaled to [0, 1] | `notebooks/01_download_images.py` |
| Laplacian pyramid | 4 levels, Gaussian blur sigma 1.0 | `src/transforms/laplacian.py` |
| DWT and SWT | Daubechies-2 (`db2`), 2 levels | `src/transforms/dwt.py` |
| `l1norm` activity window | 3 x 3 | `src/rules.py` |
| Approximation band | always averaged | `src/coeffs.py` |
| Edge preservation | L = 1.5, published constants | `src/metrics.py` |
| DenseFuse | pretrained grayscale weights, commit `4394b63` | `src/colab_setup.py` |
| Edge metric reference code | CDDFuse evaluator, commit `73681d2` | `notebooks/09_check_edge_preservation.py` |

SPECT and PET slices are shown in color by the atlas and are converted to grayscale on download,
so the functional scans are fused as luminance only.

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

The implementation is checked in `09` against the evaluation code released with CDDFuse (Zhao et al., 2023).
With that code's conventions matched, the two agree to within 0.0004 on average over the 1,560 fused images.
That code differs from mine in three conventions: it sets the weight exponent to 1 where the published value is 1.5,
it does not fold the orientation difference, and it pads the image border with zeros where mine mirrors it.
The exponent accounts for most of the remaining difference; see the last note under Results.

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
  fusion/               the four summer methods as first written, not used by the grid
notebooks/
  01_download_images.py     build the dataset
  01b_verify_dataset.py     check it before committing
  02_run_fusion_grid.py     all 15 combinations, 1560 fused images
  03_run_evaluation.py      score everything into one long CSV
  04_run_statistics.py      Friedman, Wilcoxon, Holm, effect sizes
  05_run_misalignment.py    the shift experiment, 2080 fusions
  06_rule_vs_transform.py   spreads, dominance verdicts, orderings, the post hoc set
  07_paper_figures.py       the two figures in the paper
  08_run_art_anova.py       ART ANOVA for the transform x rule interaction
  09_check_edge_preservation.py   edge metric against a published implementation
  09b_check_edge_breakdown.py     Table V breakdown, shift test at L = 1, CDDFuse code numbers
  10_run_sensitivity.py     MRI-CT rerun at 3 wavelet levels, then a 5 x 5 l1norm window
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
  art_anova.csv                   ART ANOVA, 84 rows: F, corrected p and partial eta squared
  edge_preservation_check.csv     every image scored by the reference code and four versions of mine
  edge_preservation_report.csv    the edge preservation findings under each of those scorings
  unfused_breakdown.csv           Table V's MRI share and partner term at L = 1.5 and 1
  misalignment_edge_by_power.csv  DWT and SWT shift scores at L = 1.5 and 1, 1,664 rows
  metrics_sensitivity_levels3.csv MRI-CT with both wavelets at 3 levels, 3,570 rows
  metrics_sensitivity_window5.csv MRI-CT with a 5 x 5 l1norm window, 3,570 rows
  sensitivity_summary.csv         each rerun's answers against the main grid
  sensitivity_tests.csv           04's planned tests on the main grid and both reruns, 147 rows
app.py, registration.py, resize.py    a small Streamlit page that fuses an uploaded pair, separate from the study
```


## How to run

Everything is written for Google Colab.

1. Run the startup cell from `src/colab_setup.py`. It mounts Drive, clones this repo, and puts `src/` on the path.
2. Work through the notebooks in numerical order. `01` assembles the dataset and `01b` audits it,
   `02` fills the grid, `03` measures it, `04` tests it, `05` runs the shift experiment,
   `06` derives the spreads and orderings, and `07` plots.
   `08` and `09` are the two checks added in revision and can be run in either order after `03`.
   `09b` comes last, since it reads the output of `05` and `09`.
   `10` reruns MRI-CT from the source images with one setting changed at a time, after `03` and `04`.
3. Run the finish cell from `src/colab_setup.py` to push the CSVs back here.

Every notebook expects `DATA`, `DRIVE_ROOT` and `REPO_PATH` from the startup cell. DO NOT forget the Startup cell.

Note that `07` and `09` reach past the CSVs: the second figure and the edge metric check read source images
and fused PNGs back out of Drive, so `02` has to have run in the same Drive for them to work.
The first figure and `08` need nothing but the committed results.

Requirements: Python 3, NumPy, SciPy, pandas, PyWavelets, scikit-image, Pillow, Matplotlib.
PyTorch is needed only for the DenseFuse row; the other four transforms run without it.
`08` also needs statsmodels, and the reference code that `09` clones needs OpenCV and scikit-learn.
Colab has all of these already. `requirements.txt` at the root covers the web app only.

## Results

Patients, not slices, are the unit. Slices collapse to a case mean first, so one patient contributes a single number to any given cell. 
Friedman screens each measure, then signed-rank tests run the seven planned contrasts, Holm-corrected inside their family.

**There is no single answer to which half matters more.**
Two of the seven scores, edge preservation and spatial frequency, go to the rule outright. 
This means its narrowest spread still clears the widest the transform manages.
The other five land ambiguous, both ranges overlapping, so the honest answer there is that it depends on where in the grid you are standing.

**The rules rank the same way everywhere; how far apart they sit does not.** 
Pooled over patients, every transform puts `maxabs` over `l1norm` over `average` on edge preservation.
Per patient that also applies in 163 of 175 cells, with the DWT on MRI-SPECT accounting for most of the misses. 
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

**A formal test agrees that the rule's effect depends on the transform.**
An aligned rank transform ANOVA, with patient as the repeated-measures factor, finds the transform by rule interaction
significant on all seven measures in both tested pairings. With the `average` rule left out, where four transforms
give the same image, it stays significant in thirteen of the fourteen tests.
On edge preservation and spatial frequency the rule's partial eta squared is larger than the transform's in both pairings.

**Four edge preservation results depend on the weight exponent.**
Scored with only the exponent changed to 1, the value in the CDDFuse code, most findings are unchanged:
the rule order holds in all five transforms, the rule still dominates in MRI-CT and in MRI-SPECT,
and the untouched MRI still beats all twelve configurations on both functional pairings.
Four things move. The margin by which the rule dominates narrows,
the highest cell becomes the identity under `maxabs` instead of the Laplacian,
on MRI-CT the untouched MRI rises from above 2 of the 12 configurations to above 7,
and `maxabs` wins nine of the ten patient-level comparisons against `l1norm` instead of all ten
(under the DWT on MRI-SPECT the difference is no longer significant).
`results/edge_preservation_report.csv` has the full comparison.
Removing the orientation fold on its own narrows the margin only slightly and leaves the highest cell
and the MRI-CT count where they were, but it has the same effect on the patient-level comparisons: nine of ten.
With both changed, as in the reference code, the MRI-CT count is 8,
the rule and the transform are level in MRI-SPECT, and `maxabs` wins eight of the ten.

**One result depends on the `l1norm` window.**
In a rerun of MRI-CT with one setting changed at a time, putting both wavelets at 3 levels leaves every planned comparison pointing the same way.
A 5 x 5 window turns two around. One is a tie on entropy under the Laplacian (corrected p = 0.49).
The other is the SWT against the DWT on mutual information under `l1norm`: the DWT moves ahead,
median difference -0.018 against +0.025 at 3 x 3, corrected p = 0.0137.
So the SWT's lead on the measures that look back at the sources holds at the 3 x 3 setting, but not in every case at 5 x 5.
`results/sensitivity_tests.csv` has every planned comparison for both reruns.

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

Point `DENSEFUSE_REPO` at the path of that clone.  
  
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
- J. O. Wobbrock, L. Findlater, D. Gergle, and J. J. Higgins, "The aligned rank transform for
  nonparametric factorial analyses using only ANOVA procedures," *Proc. SIGCHI Conference on Human
  Factors in Computing Systems*, pp. 143-146, 2011.
- Z. Zhao et al., "CDDFuse: Correlation-driven dual-branch feature decomposition for multi-modality
  image fusion," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023.
