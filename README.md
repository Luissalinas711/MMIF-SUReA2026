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
This project implements four fusion methods in five configurations (DenseFuse is run under two different fusion rules) applies all of them to the same brain-image pairs, and scores 150 fused images on seven quality metrics.   
Every method is the same three-step move in a different representation: transform → combine → invert. What changes between them is the basis, and whether that basis is fixed or learned.

## Methods
1. Simple Pixel Averaging - no transform at all, just the elementwise mean. This is the baseline the others are measured against.  
2. Laplacian Pyramid Fusion - Multi-scale basis. Blur and subtract into band-pass layers, keep the larger-magnitude coefficient at each scale, then invert back.  
3. Discrete Wavelet Transform (DWT) Fusion (db2, two levels) - a fixed orthogonal basis. Average the coarse approximation, max-magnitude on the detail bands.  
4. DenseFuse (Deep Learning) - a pretrained Convolutional Neural Network (CNN) autoencoder. Run under two fusion rules, addition and ℓ1-norm.  

The seven metrics for evaluating each fused image are entropy, mutual information, standard deviation, spatial frequency, SSIM to the MRI, SSIM to the partner modality, and edge preservation.  

## How to Run 
Everything in this repository is designed for Google Colab.

1. Copy a notebook from notebooks/ into a Colab notebook.  
2. Run the startup cell from src/colab_setup.py. It mounts Google Drive, clones this repo, and puts src/ on the path.  
3. Run the notebook. Fused images and comparison figures are written to results/.

Requirements: Python 3, NumPy/SciPy, PyWavelets, scikit-image, PyTorch, Matplotlib.  
Colab has most of these preinstalled; the startup cell installs the rest.  

## Results  
Three core findings:
A detail–fidelity trade-off establishes no clear winner - The Laplacian pyramid leads every sharpness and edge measure (standard deviation, spatial frequency, SSIM to the MRI, and edge preservation) while the blending methods (averaging and DenseFuse fusion via addition method) retain the most information from the second modality. No configuration performs best at either.  
The SPECT wash-out - In MRI–SPECT pairings the Laplacian pyramid reaches 0.95 SSIM to the MRI but only 0.36 to the SPECT. The method that wins every detail metric is the worst choice when trying to retain functional signal from SPECT image; blending methods preserve it best, at roughly 0.55.  
The fusion rule can matter more than the transform - Holding the DenseFuse network fixed and switching from addition to the ℓ1-norm rule gives +51% spatial frequency and +35% edge preservation, at a cost of −3% mutual information and −3% partner SSIM. Same weights, same encoder, same decoder, only the rule changed.  
  
Per-pairing evaluation metrics for every configuration are in results/  

## Notes on DenseFuse   

DenseFuse (Li & Wu, 2019) is an autoencoder trained to reconstruct ordinary images. Because no ground-truth fused images exist, it never trains on fusing images. Instead, a fusion step is inserted between the encoder and decoder phases. We use the authors' pretrained model so no training is required. The authors' repo is cloned at runtime and intentionally not committed here.  
`05a_densefuse_add.py` and `05b_densefuse_norm.py`fetches [`hli1221/densefuse-pytorch`](https://github.com/hli1221/densefuse-pytorch), which ships both the network and the pretrained grayscale weights.  

The two fusion rules compared here are: addition, which averages the two feature tensors, and the ℓ1-norm rule, which weights each source by its local feature activity. Because both share one fixed network (same weights, same encoder, same decoder) the difference between them isolates the fusion rule for comparative purposes. The two rules are run from separate notebooks, 05a_densefuse_add.py and 05b_densefuse_norm.py, writing to results/densefuse/ and results/densefuse_norm/. Note: the norm variation was a bonus addition to the overall work after many commits from all other methods, hence the naming of some DenseFuse addition files as purely 'densefuse' rather than 'densefuse_add'  

## Dataset

Harvard Whole Brain Atlas (AANLIB)
https://www.med.harvard.edu/AANLIB/home.html

Source images are stored in Google Drive. Source images will not be committed to this repository, unless alongside resulting fused image. Fused results will be stored here.
See data/README.md for the Google Drive path and any download instructions.

## References  
- H. Li and X.-J. Wu, “DenseFuse: A Fusion Approach to Infrared and Visible Images,” IEEE Transactions on Image Processing, vol. 28, no. 5, pp. 2614–2623, 2019.  
- G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, “Densely Connected Convolutional Networks,” in Proc. IEEE CVPR, 2017, pp. 4700–4708.  
- S. L. Brunton and J. N. Kutz, Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge, U.K.: Cambridge Univ. Press, 2019.  
- S. Mallat, “A Theory for Multiresolution Signal Decomposition: The Wavelet Representation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 11, no. 7, pp. 674–693, 1989.  
- Burt, P. J., & Adelson, E. H. (1983). The Laplacian pyramid as a compact image code. IEEE Transactions on Communications, 31(4), 532-540. https://doi.org/10.1109/TCOM.1983.1095851  
