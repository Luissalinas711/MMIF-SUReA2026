# The script every fusion method uses, including the function that runs a fusion
# All four methods are the same three steps:
# transform the image, combine the two images there, transform back.
# So a transform only has to know how to pull an image apart and put it back together.
# Then any rule works with any transform.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# One image after a transform pulled it apart.
# approximation is the coarse brightness band, or none if the transform does not make one.
# detail_bands is a flat list. The wavelets give back tuples per level, that i must unpack
# rebuild_info is whatever the inverse needs (like the wavelet name)
@dataclass
class Decomposition:
    approximation: Optional[np.ndarray]
    detail_bands: List[np.ndarray]
    rebuild_info: Dict[str, Any] = field(default_factory=dict)


# Combines two decompositions using whichever rule gets input

# The approximation always gets averaged no matter what the rule is, and this is the only place it happens
# My old code was inconsistent about it: 
# my old laplacian ran max magnitude on its coarsest level while my old dwt averaged its LL band
# Thus, I could not tell a transform difference from a rule difference.

# Averaging it is the right call because that band is overall brightness.
# letting a rule pick a  winner there means one scan decides how bright the whole fused image comes out
def combine_decompositions(decomposition_a, decomposition_b, rule):
    if decomposition_a.approximation is None:
        combined_approximation = None
    else:
        combined_approximation = 0.5 * (decomposition_a.approximation
                                        + decomposition_b.approximation)

    combined_bands = [rule(band_a, band_b) for band_a, band_b
                      in zip(decomposition_a.detail_bands, decomposition_b.detail_bands)]

    # rebuild_info says how the image was pulled apart, which fusing does not change.
    return Decomposition(combined_approximation, combined_bands,
                         dict(decomposition_a.rebuild_info))


# Runs one transform and rule combination on a pair of images.
# A transform is a (forward, inverse) pair from the transforms folder. A rule is a plain function.
def fuse(image_a, image_b, transform, rule):
    forward, inverse = transform

    decomposition_a = forward(image_a)
    decomposition_b = forward(image_b)

    return inverse(combine_decompositions(decomposition_a, decomposition_b, rule))


# Wraps a transform and rule into a function taking just two images
def make_fusion_function(transform, rule):
    def fuse_one_pair(image_a, image_b):
        return fuse(image_a, image_b, transform, rule)
    return fuse_one_pair
