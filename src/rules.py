# The three ways of combining two sets of coefficients. 
# Each is a plain function that takes two bands and returns one.

# They are all the same operation with different weights:
# fused = weight_a * coefficient_a + weight_b * coefficient_b,  weights add to 1

# Bands are 2D for wavelets, the pyramid and raw pixels or 3D as for DenseFuse features (channels, height, width) 

import numpy as np
from scipy.ndimage import uniform_filter

# How big a neighbourhood l1norm looks at. one setting for each each transform
ACTIVITY_WINDOW_SIZE = 3

# Keeps l1norm from dividing by zero where both sources are silent.
TINY_NUMBER = 1e-10


# Straight average, same as the old fusion/averaging.py.
def average(band_a, band_b):
    return 0.5 * (band_a + band_b)


# Keeps whichever source is stronger and throws the other away.
# The absolute value matters b/c coefficients can be negative
def maxabs(band_a, band_b):
    return np.where(np.abs(band_a) >= np.abs(band_b), band_a, band_b)


# How much is going on in each neighbourhood
def measure_local_activity(band, window_size=ACTIVITY_WINDOW_SIZE):
    magnitude = np.abs(band)

    if magnitude.ndim == 3:
        magnitude = magnitude.sum(axis=0)

    return uniform_filter(magnitude, size=window_size)


# Weights each source by how busy it is nearby, so the quieter one only fades instead of outright zeroing out
# This is the rule DenseFuse uses
def l1norm(band_a, band_b, window_size=ACTIVITY_WINDOW_SIZE):
    activity_a = measure_local_activity(band_a, window_size)
    activity_b = measure_local_activity(band_b, window_size)

    weight_a = activity_a / (activity_a + activity_b + TINY_NUMBER)

    # DenseFuse bands have channels but the weights do not, so one weight map covers them all
    if band_a.ndim == 3:
        weight_a = weight_a[None, :, :]

    return weight_a * band_a + (1.0 - weight_a) * band_b


RULES = {'average': average, 'maxabs': maxabs, 'l1norm': l1norm}


# Every rule has to return the shape it was given
# feeding it two identical bands has to give that band straight back
if __name__ == '__main__':
    random_numbers = np.random.default_rng(0)

    for shape in [(64, 64), (16, 64, 64)]:
        band_a = random_numbers.standard_normal(shape)
        band_b = random_numbers.standard_normal(shape)

        for rule_name, rule in RULES.items():
            mixed = rule(band_a, band_b)
            same_twice = rule(band_a, band_a)

            shape_result = 'ok' if mixed.shape == band_a.shape else 'FAIL'
            identical_error = float(np.abs(same_twice - band_a).max())

            print(f'  {rule_name:<8} {str(shape):<14} shape {shape_result}   '
                  f'identical inputs error {identical_error:.2e}')
