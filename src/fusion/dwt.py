# DWT fusion
# Idea: break each image into wavelet subbands (a coarse approximation and detail bands), fuse the subbands, then rebuild. 
# Wavelets work well here b/c the detail bands essentially isolate the edges at each scale.

import numpy as np
import pywt   # pywt does the actual wavelet transform


def dwt_fuse(img_A, img_B, wavelet='db2', level=2):
    # wavelet='db2' (Daubechies 2): 4 filter coefficients, 2 vanishing moments.
    # I picked db2 over Haar b/c it gives smoother reconstruction and fewer blocky edges (clearly better on medical images)
    # level=2: two rounds of decomposition. level 1 gets fine detail, level 2 gets medium detail. Going higher just costs time w/out much improvement.

    # decompose both images into subbands.
    # wavedec2 returns a list, coarsest first:
    #   [approximation, (H_n,V_n,D_n), ... , (H_1,V_1,D_1)]
    # index 0 = the coarse approximation (LL), everything after that = the (horizontal, vertical, diagonal) detail at each level.
    coeffs_A = pywt.wavedec2(img_A, wavelet, level=level)
    coeffs_B = pywt.wavedec2(img_B, wavelet, level=level)

    # fuse the approximation by averaging. 
    # both images share the same coarse overall structure, so averaging it makes sense 
    fused = [(coeffs_A[0] + coeffs_B[0]) / 2.0]

    # fuse each detail level with the max-absolute-value rule: 
    # at each pixel, keep whichever coefficient has the bigger magnitude = the stronger edge.
    # np.abs matters here b/c detail coefficients can be + or -.
    # Example: -0.8 is a strong edge, but plain np.maximum would throw it out
    # for a weak +0.3. np.abs compares strength, np.where keeps the real sign.
    for (hA, vA, dA), (hB, vB, dB) in zip(coeffs_A[1:], coeffs_B[1:]):
        fused.append((
            np.where(np.abs(hA) >= np.abs(hB), hA, hB),   # horizontal edges
            np.where(np.abs(vA) >= np.abs(vB), vA, vB),   # vertical edges
            np.where(np.abs(dA) >= np.abs(dB), dA, dB),   # diagonal detail
        ))

    # rebuild the fused image from the combined subbands
    recon = pywt.waverec2(fused, wavelet)
    # db2 sometimes hands back an image 1 pixel bigger per side (the filter length makes the size round up), so crop back to the original size.
    # clip at the end b/c tiny floating point errors can drift out of [0,1].
    return np.clip(recon[:img_A.shape[0], :img_A.shape[1]], 0.0, 1.0)
