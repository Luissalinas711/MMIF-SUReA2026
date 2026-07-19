# Fusion quality metrics.
# No ground-truth fused image exists, so quality is determined by how well the fused image keeps information from both sources.
# Every function takes grayscale numpy images in [0,1]. MRI and partner are the two sources; fused is the result.
# Note: partner here just means the second modality source (CT, PET, or SPECT)

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel


def entropy(image, bins=256):
    # How much information/detail the image holds (Higher = more)
    # histogram -> probabilities
    counts, _ = np.histogram(image, bins=bins, range=(0, 1))
    prob = counts / counts.sum()  
    # drop empty bins so log2 stays finite
    prob = prob[prob > 0]          
    return float(-np.sum(prob * np.log2(prob)))

def _pair_mi(image_a, image_b, bins=256):
    # Mutual information between two images from their joint histogram.
    joint, _, _ = np.histogram2d(image_a.ravel(), image_b.ravel(),
                                 bins=bins, range=[[0, 1], [0, 1]])
    joint_prob = joint / joint.sum()          # p(a, b)
    prob_a = joint_prob.sum(axis=1)           # p(a)
    prob_b = joint_prob.sum(axis=0)           # p(b)

    independent = prob_a[:, None] * prob_b[None, :]     
    nonzero = joint_prob > 0                            # only sum where p(a,b) > 0
    ratio = joint_prob[nonzero] / independent[nonzero]
    return float(np.sum(joint_prob[nonzero] * np.log2(ratio)))

def mutual_information(mri, partner, fused, bins=256):
    # How much of each source survives in the fused image. Higher = more preserved.
    return _pair_mi(mri, fused, bins) + _pair_mi(partner, fused, bins)

def standard_deviation(image):
    # Contrast shows how spread out the intensities are.
    return float(np.std(image))

def spatial_frequency(image):
    # Sharpness/fine detail, from the row and column gradients.
    row_freq = np.sqrt(np.mean(np.diff(image, axis=1) ** 2))
    col_freq = np.sqrt(np.mean(np.diff(image, axis=0) ** 2))
    return float(np.sqrt(row_freq ** 2 + col_freq ** 2))

def ssim_to_sources(mri, partner, fused):
    # Structural similarity of the fused image to each source (higher = closer).
    ssim_mri = ssim(mri, fused, data_range=1.0)
    ssim_partner = ssim(partner, fused, data_range=1.0)
    return float(ssim_mri), float(ssim_partner)

def all_metrics(mri, partner, fused):
    ssim_mri, ssim_partner = ssim_to_sources(mri, partner, fused)
    return {
        'entropy':      entropy(fused),
        'MI':           mutual_information(mri, partner, fused),
        'std':          standard_deviation(fused),
        'spatial_freq': spatial_frequency(fused),
        'SSIM_mri':     ssim_mri,
        'SSIM_partner': ssim_partner,
    }
