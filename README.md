# Multimodal Medical Image Fusion - SUReA 2026
SUReA 2026 undergraduate research at CSUF comparing four multimodal medical image fusion methods (Simple Averaging, Laplacian Pyramids, Discrete Wavelet Transforms, DenseFuse) on Harvard Whole Brain Atlas medical images.

**Student:** Luis Rey Salinas Jr.  
**Mentor:** Dr. Yoonsuk Choi  
**School:** California State University, Fullerton  
**Program:** Summer Undergraduate Research Academy, 2026  

## Overview  
No single medical scanner captures everything a clinician needs.
MRI resolves soft tissue,CT resolves bone, and PET/SPECT show metabolic or perfusion activity.
Mulitmodal medical image fusion combines two registered scans into a single image that keeps the useful detail of both.  
This project implements four fusion methods, runs them all on the same brain-image pairs from the Harvard Whole Brain Atlas, and measures how well each preserves information from both sources.  

## Status

Week 6 - DenseFuse Fusion

## Methods

1. Simple Averaging
2. Laplacian Pyramid Fusion
3. Discrete Wavelet Transform (DWT) Fusion
4. DenseFuse (Deep Learning)  

## Notes on DenseFuse (Method 4)  

DenseFuse (Li & Wu, 2019) is an autoencoder trained to reconstruct ordinary images.  
Because no ground-truth fused images exist, it never trains on fusing images.  
instead, a fusion step is inserted between the encoder and decoder phases.  
We use the authors' pretrained model so no training is required.  
The authors' repo is cloned at runtime and intentionally not committed here.  
`05_densefuse.py` fetches [`hli1221/densefuse-pytorch`](https://github.com/hli1221/densefuse-pytorch), which ships both the network and the pretrained grayscale weights.


## Dataset

Harvard Whole Brain Atlas (AANLIB)
https://www.med.harvard.edu/AANLIB/home.html

Source images are stored in Google Drive. Source images will not be committed to this repository, unless alongside resulting fused image. Fused results will be stored here.
See data/README.md for the Google Drive path and any download instructions.

## Executing Scripts  
Everything in this repository is designed for Google Colab.

1. Copy/paste a notebook from `notebooks/` into your Colab environment.  
2. Run the startup cell. This mounts Google Drive, clones this repo, adds `src/` to the path.  
3. Run the notebook. Fused images and comparison figures are written to `results/`.  

Requirements: Python 3, NumPy/SciPy, PyWavelets, scikit-image, PyTorch, Matplotlib.  
Colab has most of these preinstalled; the startup cell installs the rest.  
