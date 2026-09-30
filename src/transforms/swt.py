# Method 4. SWT, 
# the same as the DWT without the downsampling, which is what makes it shift invariant. 

import numpy as np

from transforms.dwt import (WAVELET_NAME, WAVELET_LEVELS, crop_and_clip,
                            pack_wavelet, unpack_wavelet)

def forward(image, wavelet=WAVELET_NAME, levels=WAVELET_LEVELS):
    import pywt
    pixels = np.asarray(image, dtype=np.float64)
    coefficients = pywt.swt2(pixels, wavelet, level=levels, trim_approx=True, norm=True)

    return pack_wavelet(coefficients, pixels.shape, wavelet, levels)


def inverse(decomposition):
    import pywt
    rebuilt = pywt.iswt2([decomposition.approximation] + unpack_wavelet(decomposition),
                         decomposition.rebuild_info['wavelet'], norm=True)
    return crop_and_clip(rebuilt, decomposition.rebuild_info['shape'])
