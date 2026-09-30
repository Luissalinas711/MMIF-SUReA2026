# Collects the five transforms so pipeline.py can look one up by name.
# Each one is a script in this folder with a forward and an inverse,
from . import averaging, laplacian, dwt, swt, densefuse

import numpy as np

# each entry is a (forward, inverse) pair, which is what coeffs.fuse needs
TRANSFORMS = {
    'averaging': (averaging.forward, averaging.inverse),
    'laplacian': (laplacian.forward, laplacian.inverse),
    'dwt': (dwt.forward, dwt.inverse),
    'swt': (swt.forward, swt.inverse),
    'densefuse': (densefuse.forward, densefuse.inverse),
}

# DenseFuse needs torch and its authors' cloned repo, so this list helps with quick checks
CLASSICAL_TRANSFORMS = ['averaging', 'laplacian', 'dwt', 'swt']


# Every transform has to rebuild an image it just pulled apart
def check_round_trip(tolerance=1e-10):
    test_image = np.random.default_rng(0).random((256, 256))
    errors = {}

    print('round trip check:')
    for transform_name in CLASSICAL_TRANSFORMS:
        forward, inverse = TRANSFORMS[transform_name]

        decomposition = forward(test_image)
        error = float(np.abs(inverse(decomposition) - test_image).max())
        errors[transform_name] = error

        result = 'ok  ' if error < tolerance else 'FAIL'
        band_count = len(decomposition.detail_bands)
        has_approximation = 'yes' if decomposition.approximation is not None else 'no'

        print(f'  {result} {transform_name:<10} largest error {error:.2e}   '
              f'detail bands {band_count:<2} approximation {has_approximation}')

    worst_error = max(errors.values())
    verdict = 'all good' if worst_error < tolerance else 'something is wrong'
    print(f'\nworst error {worst_error:.2e}, {verdict}')

    return errors
