
# Fusion quality metrics.
# There is no ground truth fused image, so quality is judged by how well the fused image keeps information from both sources.
# Every function takes grayscale numpy images in [0,1]. 
# mri and partner are the two sources, fused is the result
# Partner is just the second modality: CT, PET, or SPECT.

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel

TINY_NUMBER = 1e-10


def entropy(image, bins=256):
    # How much information the image holds. Higher is more.
    counts, _ = np.histogram(image, bins=bins, range=(0, 1))
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]   
    return float(-np.sum(probabilities * np.log2(probabilities)))

# Mutual information between two images, from their joint histogram.
def mutual_information_pair(image_a, image_b, bins=256):
    joint_counts, _, _ = np.histogram2d(image_a.ravel(), image_b.ravel(),
                                        bins=bins, range=[[0, 1], [0, 1]])
    joint_probability = joint_counts / joint_counts.sum()

    probability_a = joint_probability.sum(axis=1)
    probability_b = joint_probability.sum(axis=0)

    expected_if_independent = probability_a[:, None] * probability_b[None, :]
    nonzero = joint_probability > 0
    ratio = joint_probability[nonzero] / expected_if_independent[nonzero]
    return float(np.sum(joint_probability[nonzero] * np.log2(ratio)))

# How much of each source survives in the fused image. Higher is more preserved.
def mutual_information(mri, partner, fused, bins=256):
    return (mutual_information_pair(mri, fused, bins)
            + mutual_information_pair(partner, fused, bins))

# Contrast: meaning how spread out the intensities are
def standard_deviation(image):
    return float(np.std(image))

# Sharpness and fine detail, from row and column gradients
def spatial_frequency(image):
    row_frequency = np.sqrt(np.mean(np.diff(image, axis=1) ** 2))
    column_frequency = np.sqrt(np.mean(np.diff(image, axis=0) ** 2))
    return float(np.sqrt(row_frequency ** 2 + column_frequency ** 2))

# Structural similarity of the fused image to each source. Higher is closer to source
def ssim_to_sources(mri, partner, fused):

    return float(ssim(mri, fused, data_range=1.0)), float(ssim(partner, fused, data_range=1.0))


# Constants from Xydeas and Petrovic (2000). These are the published values
# The two sigmoids decide how forgiving the score is. 
# Strength is judged around a ratio of 0.5 and orientation around a match of 0.8
# both drop off sharply past that
STRENGTH_GAMMA = 0.9994
STRENGTH_KAPPA = -15.0
STRENGTH_SIGMA = 0.5

ORIENTATION_GAMMA = 0.9879
ORIENTATION_KAPPA = -22.0
ORIENTATION_SIGMA = 0.8

# Each pixel counts in proportion to how strong that source's edge is there, raised to this power.
# 1.5 is what is used by Xydeas and Petrović
EDGE_WEIGHT_POWER = 1.5

# Edge strength and orientation at every pixel.
def sobel_edges(image):
    gradient_x = sobel(image, axis=1)
    gradient_y = sobel(image, axis=0)
    strength = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    orientation = np.arctan2(gradient_y, gradient_x)
    return strength, orientation


def edges_kept(source_strength, source_orientation, fused_strength, fused_orientation):
    # How well one source's edges survive in the fused image, 0 to 1 per pixel.

    # How close the fused edge strength is to the source's. Always the weaker over the stronger, so
    # it does not matter which one is bigger.
    weaker_edge = np.minimum(fused_strength, source_strength)
    stronger_edge = np.maximum(fused_strength, source_strength) + TINY_NUMBER
    strength_ratio = weaker_edge / stronger_edge

    # Angle between the two edges
    # arctan2 gives angles in (-pi, pi], so the difference between an edge at -pi and one at +pi reads as almost 2pi 
    # but they actually point the same way. Wrapping is used to fix that
    # Also, an edge has no direction. A boundary that goes dark to light or light to dark is still the same boundary
    # Thus angles pi apart count as a match too.
    orientation_gap = np.abs(source_orientation - fused_orientation) % (2.0 * np.pi)
    orientation_gap = np.minimum(orientation_gap, 2.0 * np.pi - orientation_gap)   # now [0, pi]
    orientation_gap = np.minimum(orientation_gap, np.pi - orientation_gap)         # now [0, pi/2]
    orientation_match = 1.0 - orientation_gap / (np.pi / 2.0)

    strength_quality = STRENGTH_GAMMA / (
        1.0 + np.exp(STRENGTH_KAPPA * (strength_ratio - STRENGTH_SIGMA)))
    orientation_quality = ORIENTATION_GAMMA / (
        1.0 + np.exp(ORIENTATION_KAPPA * (orientation_match - ORIENTATION_SIGMA)))

    return strength_quality * orientation_quality


def edge_preservation(mri, partner, fused):
    # The Xydeas and Petrovic edge preservation measure, written Q^AB/F.
    # Compares the fused image's edges against each source, 
    # weights every pixel by how strong that source's edge is there, and combines. 
    # about 0 to 1, higher means edges are better kept.
    mri_strength, mri_orientation = sobel_edges(mri)
    partner_strength, partner_orientation = sobel_edges(partner)
    fused_strength, fused_orientation = sobel_edges(fused)

    kept_from_mri = edges_kept(mri_strength, mri_orientation,
                               fused_strength, fused_orientation)
    kept_from_partner = edges_kept(partner_strength, partner_orientation,
                                   fused_strength, fused_orientation)

    mri_weight = mri_strength ** EDGE_WEIGHT_POWER
    partner_weight = partner_strength ** EDGE_WEIGHT_POWER

    weighted_total = np.sum(kept_from_mri * mri_weight + kept_from_partner * partner_weight)
    total_weight = np.sum(mri_weight + partner_weight) + TINY_NUMBER

    return float(weighted_total / total_weight)


def all_metrics(mri, partner, fused):
    ssim_mri, ssim_partner = ssim_to_sources(mri, partner, fused)
    return {
        'entropy': entropy(fused),
        'MI': mutual_information(mri, partner, fused),
        'std': standard_deviation(fused),
        'spatial_freq': spatial_frequency(fused),
        'SSIM_mri': ssim_mri,
        'SSIM_partner': ssim_partner,
        'edge_preservation': edge_preservation(mri, partner, fused),
    }
