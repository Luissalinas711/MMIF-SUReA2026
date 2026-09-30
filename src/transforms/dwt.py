# Method 3: DWT 

import numpy as np

from coeffs import Decomposition

# The DWT and SWT use the same wavelet at the same depth on purpose. 
# The reason to add the SWT is to isolate what the downsampling costs
WAVELET_NAME = 'db2'
WAVELET_LEVELS = 2


# pywt gives back the coarse band first, then a horizontal, vertical and diagonal tuple per level.
# we 'flatten' the tuples
def pack_wavelet(coefficients, shape, wavelet, levels):
    detail_bands = []
    for band_group in coefficients[1:]:
        detail_bands.extend(band_group)

    return Decomposition(coefficients[0], detail_bands,
                         {'wavelet': wavelet, 'levels': levels, 'shape': shape})


# Groups the flat list back into threes (the desired shape for rebuidling)
def unpack_wavelet(decomposition):
    bands = decomposition.detail_bands
    return [tuple(bands[start:start + 3]) for start in range(0, len(bands), 3)]


# Trims a rebuilt image back to size and into range. Wavelet rebuilds can come back a pixel bigger per side.  
# also, floating point drift can push values slightly outside [0, 1].
def crop_and_clip(image, shape):
    height, width = shape
    return np.clip(image[:height, :width], 0.0, 1.0)


def forward(image, wavelet=WAVELET_NAME, levels=WAVELET_LEVELS):
    import pywt

    pixels = np.asarray(image, dtype=np.float64)
    coefficients = pywt.wavedec2(pixels, wavelet, level=levels)

    return pack_wavelet(coefficients, pixels.shape, wavelet, levels)


def inverse(decomposition):
    import pywt

    rebuilt = pywt.waverec2([decomposition.approximation] + unpack_wavelet(decomposition),
                            decomposition.rebuild_info['wavelet'])

    return crop_and_clip(rebuilt, decomposition.rebuild_info['shape'])
