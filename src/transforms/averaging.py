# Method 1: No transform at all, so the rules act straight on raw pixels.

import numpy as np

from coeffs import Decomposition

def forward(image):
    return Decomposition(None, [np.asarray(image, dtype=np.float64)], {})

def inverse(decomposition):
    return np.clip(decomposition.detail_bands[0], 0.0, 1.0)
