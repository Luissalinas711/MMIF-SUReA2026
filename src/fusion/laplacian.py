# Laplacian pyramid fusion.
# This one file cleanly replaces my two old notebooks, as they were the same function and may have caused confusion. 
# It uses scipy's gaussian_filter for the blur (fast), but does an upsample by inserting zeros so the reconstruction comes out right.

# The idea is to build a Gaussian pyramid (blur + shrink over and over), 
# turn that into a Laplacian pyramid (the detail lost at each blur step), 
# fuse the two images level by level, 
# then add it all back up.

import numpy as np
from scipy.ndimage import gaussian_filter


# Reduce step: blur, then throw away every other pixel (shrink by 2)
def _reduce(img, sigma):
    return gaussian_filter(img, sigma=sigma)[::2, ::2]


# Expand step: grow back by 2. Insert a zero between every pixel, blur to fill the gaps, then scale by 4. 
# The x4 is b/c after inserting zeros only ~1/4 of the pixels are non-zero, so the blur averages in a bunch of zeros and comes out too dark w/out it.
def _expand(img, shape, sigma):
    H, W = img.shape
    up = np.zeros((H * 2, W * 2), dtype=np.float32)
    up[::2, ::2] = img                       # original pixels go at the even spots
    up = gaussian_filter(up, sigma=sigma) * 4.0
    return up[:shape[0], :shape[1]]          # crop to the exact size we need


# Gaussian pyramid = the image at shrinking sizes (each one blurred + reduced)
def _gaussian_pyramid(img, levels, sigma):
    pyr = [img.astype(np.float32)]
    for _ in range(levels - 1):
        pyr.append(_reduce(pyr[-1], sigma))
    return pyr


# Laplacian pyramid = the detail at each level.
# detail = this level minus the expanded version of the next (coarser) level,
# i.e. what got lost in the blur/shrink. 
# The very last (coarsest) level is kept as is to be the base we rebuild from.
def _laplacian_pyramid(gp, sigma):
    lap = [gp[k] - _expand(gp[k + 1], gp[k].shape, sigma)
           for k in range(len(gp) - 1)]
    lap.append(gp[-1])
    return lap


def laplacian_pyramid_fuse(img_A, img_B, levels=4, sigma=1.0):
    # levels=4 gave the best balance of detail vs speed on the 256x256 images
    lap_A = _laplacian_pyramid(_gaussian_pyramid(img_A, levels, sigma), sigma)
    lap_B = _laplacian_pyramid(_gaussian_pyramid(img_B, levels, sigma), sigma)

    # fuse level by level: at each pixel keep whichever detail is stronger (bigger magnitude). Same max-abs idea as the DWT method.
    # Note: this also picks on the coarsest/base level. Averaging the base instead (like I do for the DWT approximation) might be cleaner for brightness
    # but leaving it as max-abs for now so it matches what I had.
    fused = [np.where(np.abs(a) >= np.abs(b), a, b)
             for a, b in zip(lap_A, lap_B)]

    # rebuild: start from the coarsest fused level, then expand and add each finer detail level back on top
    out = fused[-1]
    for detail in reversed(fused[:-1]):
        out = _expand(out, detail.shape, sigma) + detail
    return np.clip(out, 0.0, 1.0)
