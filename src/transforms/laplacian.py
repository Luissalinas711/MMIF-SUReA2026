# Method 2: Laplacian pyramid.

import numpy as np
from scipy.ndimage import gaussian_filter

from coeffs import Decomposition

LEVELS = 4
BLUR_SIGMA = 1.0

# Blur, then drop every other row and column so the image halves.
def blur_and_shrink(image, sigma):
    return gaussian_filter(image, sigma=sigma)[::2, ::2]

# The opposite. Put a zero between every pixel, blur to fill the gaps, multiply by 4. 
# The multiply is needed b/c only about a quarter of the pixels have anything in them after inserting zeros,
# so the blur averages in all those zeros and comes out too dark.
def grow_and_blur(image, target_shape, sigma):
    height, width = image.shape

    grown = np.zeros((height * 2, width * 2), dtype=image.dtype)
    grown[::2, ::2] = image
    grown = gaussian_filter(grown, sigma=sigma) * 4.0

    return grown[:target_shape[0], :target_shape[1]]

def forward(image, levels=LEVELS, sigma=BLUR_SIGMA):
    # the image at shrinking sizes
    sizes = [np.asarray(image, dtype=np.float64)]
    for _ in range(levels - 1):
        sizes.append(blur_and_shrink(sizes[-1], sigma))

    # each detail band is what got lost going from one size to the next
    detail_bands = []
    for level in range(len(sizes) - 1):
        finer = sizes[level]
        detail_bands.append(finer - grow_and_blur(sizes[level + 1], finer.shape, sigma))

    return Decomposition(sizes[-1], detail_bands, {'sigma': sigma, 'levels': levels})


def inverse(decomposition):
    sigma = decomposition.rebuild_info['sigma']

    # start from the smallest version and add each detail band back on the way up
    rebuilt = decomposition.approximation
    for detail_band in reversed(decomposition.detail_bands):
        rebuilt = grow_and_blur(rebuilt, detail_band.shape, sigma) + detail_band

    return np.clip(rebuilt, 0.0, 1.0)
