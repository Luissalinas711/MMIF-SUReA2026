# Fusion quality metrics.
# No ground-truth fused image exists, so quality is determined by how well the fused image keeps information from both sources.
# Every function takes grayscale numpy images in [0,1]. MRI and partner are the two sources; fused is the result.
# Note: partner here just means the second modality source (CT, PET, or SPECT)

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel


def entropy(image, bins=256):
    # How much information/detail the image holds (higher = more)
    counts, _ = np.histogram(image, bins=bins, range=(0, 1))
    probabilities = counts / counts.sum()          # histogram -> probabilities
    probabilities = probabilities[probabilities > 0]   # drop empty bins so log2 stays finite
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mutual_info_pair(image_a, image_b, bins=256):
    # Mutual information between two images, from their joint histogram.
    joint_counts, _, _ = np.histogram2d(image_a.ravel(), image_b.ravel(),
                                         bins=bins, range=[[0, 1], [0, 1]])
    joint_prob = joint_counts / joint_counts.sum()     # p(a, b)
    prob_a = joint_prob.sum(axis=1)                    # p(a)
    prob_b = joint_prob.sum(axis=0)                    # p(b)

    expected_if_independent = prob_a[:, None] * prob_b[None, :]     # p(a) * p(b)
    nonzero = joint_prob > 0                            # only sum where p(a,b) > 0
    ratio = joint_prob[nonzero] / expected_if_independent[nonzero]
    return float(np.sum(joint_prob[nonzero] * np.log2(ratio)))


def mutual_information(mri, partner, fused, bins=256):
    # How much of each source survives in the fused image. Higher = more preserved.
    return _mutual_info_pair(mri, fused, bins) + _mutual_info_pair(partner, fused, bins)


def standard_deviation(image):
    # Contrast: how spread out the intensities are.
    return float(np.std(image))


def spatial_frequency(image):
    # Sharpness / fine detail, from the row and column gradients.
    row_frequency = np.sqrt(np.mean(np.diff(image, axis=1) ** 2))
    column_frequency = np.sqrt(np.mean(np.diff(image, axis=0) ** 2))
    return float(np.sqrt(row_frequency ** 2 + column_frequency ** 2))


def ssim_to_sources(mri, partner, fused):
    # Structural similarity of the fused image to each source (higher = closer)
    ssim_mri = ssim(mri, fused, data_range=1.0)
    ssim_partner = ssim(partner, fused, data_range=1.0)
    return float(ssim_mri), float(ssim_partner)


def _sobel_edges(image):
    # Sobel edge strength and orientation at every pixel
    gradient_x = sobel(image, axis=1)
    gradient_y = sobel(image, axis=0)
    edge_strength = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    edge_orientation = np.arctan2(gradient_y, gradient_x)
    return edge_strength, edge_orientation


def edge_preservation(mri, partner, fused):
    # How well the fused image keeps each source's edges, weighted by edge strength
    # (~0..1). Simplified version of the Xydeas-Petrovic Q^AB/F measure.
    mri_strength, mri_orientation = _sobel_edges(mri)
    partner_strength, partner_orientation = _sobel_edges(partner)
    fused_strength, fused_orientation = _sobel_edges(fused)
    epsilon = 1e-10   # small number to avoid dividing by zero

    def fraction_kept(source_strength, source_orientation):
        # How well one source's edges survive in the fused image (0 to 1 per pixel)
        weaker_edge = np.minimum(fused_strength, source_strength)
        stronger_edge = np.maximum(fused_strength, source_strength) + epsilon
        strength_match = weaker_edge / stronger_edge
        orientation_match = 1.0 - np.abs(source_orientation - fused_orientation) / (np.pi / 2.0)
        return strength_match * np.clip(orientation_match, 0, 1)

    kept_from_mri = fraction_kept(mri_strength, mri_orientation)
    kept_from_partner = fraction_kept(partner_strength, partner_orientation)

    # weight each pixel by that source's edge strength, then normalise
    weighted_sum = np.sum(kept_from_mri * mri_strength + kept_from_partner * partner_strength)
    total_strength = np.sum(mri_strength + partner_strength) + epsilon
    return float(weighted_sum / total_strength)


def all_metrics(mri, partner, fused):
    ssim_mri, ssim_partner = ssim_to_sources(mri, partner, fused)
    return {
        'entropy':          entropy(fused),
        'MI':               mutual_information(mri, partner, fused),
        'std':              standard_deviation(fused),
        'spatial_freq':     spatial_frequency(fused),
        'SSIM_mri':         ssim_mri,
        'SSIM_partner':     ssim_partner,
        'edge_preservation': edge_preservation(mri, partner, fused),
    }
