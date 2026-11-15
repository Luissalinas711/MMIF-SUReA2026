# Compares my edge preservation measure with a published implementation:
# Evaluator.Qabf in utils/Evaluator.py of the CDDFuse repository (Zhao et al., CVPR 2023).
# Run after 02. Reads the sources and the fused images, fuses nothing.

# The two differ in three ways (so the scores are not expected to be identical):
#   1. edge weight power: theirs is 1, mine is 1.5
#   2. orientation: they subtract two angles without wrapping the difference, I fold it
#   3. borders: their Sobel pads with zeros, mine mirrors the image
# Their code expects images in 0 to 255, whereas mine in 0 to 1.

# Each image is scored five ways:
#   reference - their code, unchanged
#   mine_matched - my measure with their three conventions, to check the implementation
#   mine_power_1 - my measure with only the power changed to 1
#   mine_no_fold - my measure with only the orientation fold removed
#   mine_published - my measure as used in the paper

import importlib.util
import os
import numpy as np
import pandas as pd
from scipy.ndimage import sobel
from scipy.stats import spearmanr
import metrics
from evaluation import load_grayscale, pair_names, split_method
from pipeline import MODALITIES, MRI_TOKEN
from utils import load_image_pair

# Pinned to one commit, like DenseFuse in the startup cell
REFERENCE_URL = 'https://github.com/Zhaozixiang1228/MMIF-CDDFuse'
REFERENCE_COMMIT = '73681d28f6100461f1b214ecf77a93b495be227b'
REFERENCE_REPO = '/content/MMIF-CDDFuse'

SCORE_COLUMNS = ['reference', 'mine_matched', 'mine_power_1', 'mine_no_fold', 'mine_published']
PAIRINGS = [('mri_ct', 'MRI-CT'), ('mri_spect', 'MRI-SPECT'), ('mri_pet', 'MRI-PET')]

# These three cells repeat averaging__average, so twelve of the fifteen are distinct (same as 04)
DUPLICATE_CELLS = [('laplacian', 'average'), ('dwt', 'average'), ('swt', 'average')]

# Clone their repository if it is not there yet
def fetch_reference(repo_path=REFERENCE_REPO):
    evaluator_file = f'{repo_path}/utils/Evaluator.py'
    if not os.path.isfile(evaluator_file):
        os.system(f'git clone {REFERENCE_URL} {repo_path}')
        os.system(f'git -C {repo_path} checkout {REFERENCE_COMMIT}')
    if not os.path.isfile(evaluator_file):
        raise RuntimeError(f'could not find {evaluator_file}, check the clone')
    return evaluator_file

# Loaded by file path, b/c "utils" is already the name of my own src/utils.py
def load_reference(evaluator_file):
    spec = importlib.util.spec_from_file_location('cddfuse_evaluator', evaluator_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Evaluator

# Back from [0, 1] to the 0 to 255 values of the 8 bit file
def to_255(image):
    return np.round(image * 255.0).astype(np.float32)

# Edge strength and orientation with their arctan of the slope.
# border is 'constant' for their zero padding, 'reflect' for my mirrored border
def edges_reference_style(image, border):
    gradient_x = sobel(image, axis=1, mode=border)
    gradient_y = sobel(image, axis=0, mode=border)
    strength = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # a vertical edge (no x gradient) gets pi/2
    safe_x = np.where(gradient_x == 0, 1.0, gradient_x)
    orientation = np.where(gradient_x == 0, np.pi / 2.0, np.arctan(gradient_y / safe_x))
    return strength, orientation

# Same as edges_kept in metrics.py, except the angle difference is not wrapped
def edges_kept_reference_style(source_edges, fused_edges):
    source_strength, source_orientation = source_edges
    fused_strength, fused_orientation = fused_edges
    weaker_edge = np.minimum(fused_strength, source_strength)
    stronger_edge = np.maximum(fused_strength, source_strength) + metrics.TINY_NUMBER
    strength_ratio = weaker_edge / stronger_edge
    angle_gap = np.abs(source_orientation - fused_orientation)
    orientation_match = 1.0 - angle_gap / (np.pi / 2.0)
    strength_quality = metrics.STRENGTH_GAMMA / (
        1.0 + np.exp(metrics.STRENGTH_KAPPA * (strength_ratio - metrics.STRENGTH_SIGMA)))
    orientation_quality = metrics.ORIENTATION_GAMMA / (
        1.0 + np.exp(metrics.ORIENTATION_KAPPA * (orientation_match - metrics.ORIENTATION_SIGMA)))
    return strength_quality * orientation_quality

# My measure without the orientation fold, at a chosen border and edge weight power.
# Zero padding and power 1 give all three of their conventions
def edge_preservation_no_fold(mri, partner, fused, border, power):
    mri_edges = edges_reference_style(mri, border)
    partner_edges = edges_reference_style(partner, border)
    fused_edges = edges_reference_style(fused, border)
    kept_from_mri = edges_kept_reference_style(mri_edges, fused_edges)
    kept_from_partner = edges_kept_reference_style(partner_edges, fused_edges)
    mri_weight = mri_edges[0] ** power
    partner_weight = partner_edges[0] ** power
    weighted_total = np.sum(kept_from_mri * mri_weight + kept_from_partner * partner_weight)
    total_weight = np.sum(mri_weight + partner_weight) + metrics.TINY_NUMBER
    return float(weighted_total / total_weight)

# My measure as in metrics.py, at another edge weight power. The setting is restored afterwards
def edge_preservation_at_power(mri, partner, fused, power):
    saved = metrics.EDGE_WEIGHT_POWER
    metrics.EDGE_WEIGHT_POWER = power
    score = metrics.edge_preservation(mri, partner, fused)
    metrics.EDGE_WEIGHT_POWER = saved
    return score

def score_five_ways(evaluator, mri, partner, fused):
    published_power = metrics.EDGE_WEIGHT_POWER
    return {
        'reference': float(evaluator.Qabf(to_255(fused), to_255(mri), to_255(partner))),
        'mine_matched': edge_preservation_no_fold(mri, partner, fused, 'constant', 1.0),
        'mine_power_1': edge_preservation_at_power(mri, partner, fused, 1.0),
        'mine_no_fold': edge_preservation_no_fold(mri, partner, fused, 'reflect', published_power),
        'mine_published': metrics.edge_preservation(mri, partner, fused),
    }

# Every fused image in the grid, plus the two unfused sources of each pair. Uses same walk as score_everything in evaluation.py
def score_all(data_root, results_root, evaluator):
    combinations = sorted(name for name in os.listdir(results_root) if '__' in name)
    print(f'{len(combinations)} combinations found')

    rows = []
    for subfolder, partner_token, modality in MODALITIES:
        folder = f'{data_root}/{subfolder}'
        if not os.path.isdir(folder):
            print(f'  [skip] {folder} not found')
            continue
        mri_files = sorted(f for f in os.listdir(folder)
                           if MRI_TOKEN in f and f.endswith('.png'))

        for mri_file in mri_files:
            partner_file = mri_file.replace(MRI_TOKEN, partner_token)
            mri, partner = load_image_pair(f'{folder}/{mri_file}', f'{folder}/{partner_file}')
            case, slice_number, fused_name = pair_names(mri_file, modality.lower())

            # (transform, rule, image) for each fused result, then for the two sources
            images = []
            for combination in combinations:
                fused_path = f'{results_root}/{combination}/fused_only/{fused_name}'
                if os.path.exists(fused_path):
                    transform, rule, shift = split_method(combination)
                    images.append((transform, rule, load_grayscale(fused_path)))
            images.append(('unfused', 'mri', mri))
            images.append(('unfused', 'partner', partner))
            for transform, rule, image in images:
                row = {'pairing': subfolder, 'case': case, 'slice': slice_number,
                       'transform': transform, 'rule': rule}
                row.update(score_five_ways(evaluator, mri, partner, image))
                rows.append(row)
        print(f'  {subfolder}: {len(mri_files)} pairs scored')
    return pd.DataFrame(rows)

# Patient level cell means for one score column, like Table II: slices averaged within a patient, then patients averaged
def cell_means(scores, column):
    keys = ['transform', 'rule']
    per_patient = scores.groupby(keys + ['pairing', 'case'], as_index=False)[column].mean()
    return per_patient.groupby(keys)[column].mean()

# The report lines for one score column
def report_column(scores, column):
    fused_images = scores[scores['transform'] != 'unfused']
    difference = (fused_images[column] - fused_images['reference']).abs()

    # per image pair: do both rank its configurations in the same order?
    rank_correlations = []
    for _, block in fused_images.groupby(['pairing', 'case', 'slice']):
        rank_correlations.append(spearmanr(block[column], block['reference'])[0])
    mean_rank_correlation = sum(rank_correlations) / len(rank_correlations)
    means = cell_means(scores, column)
    wide = means.drop('unfused', level='transform').unstack('rule')
    ordered = (wide['maxabs'] > wide['l1norm']) & (wide['l1norm'] > wide['average'])
    smallest_rule_spread = (wide.max(axis=1) - wide.min(axis=1)).min()
    largest_transform_spread = (wide.max(axis=0) - wide.min(axis=0)).max()
    best_transform, best_rule = wide.stack().idxmax()

    lines = {
        'mean score, fused images': f'{fused_images[column].mean():.4f}',
        'mean difference from reference': f'{difference.mean():.4f}',
        'largest difference from reference': f'{difference.max():.4f}',
        'mean rank correlation with reference': f'{mean_rank_correlation:.4f}',
        'transforms with maxabs > l1norm > average': f'{int(ordered.sum())} of {len(wide)}',
        'smallest rule spread': f'{smallest_rule_spread:.4f}',
        'largest transform spread': f'{largest_transform_spread:.4f}',
        'rule dominates': 'yes' if smallest_rule_spread > largest_transform_spread else 'no',
        'highest cell': f'{best_transform}, {best_rule}',
    }

    # unfused MRI against the twelve distinct configurations, one pairing at a time
    for pairing, label in PAIRINGS:
        means = cell_means(scores[scores['pairing'] == pairing], column)
        fused = means.drop('unfused', level='transform').drop(DUPLICATE_CELLS, errors='ignore')
        above = int((fused < means[('unfused', 'mri')]).sum())
        lines[f'unfused MRI above, {label}'] = f'{above} of {len(fused)}'
    return lines

if __name__ == '__main__':
    evaluator = load_reference(fetch_reference())
    scores = score_all(DATA, f'{DRIVE_ROOT}/results_grid', evaluator)
    scores.to_csv(f'{DRIVE_ROOT}/edge_preservation_check.csv', index=False)
    print(f'{len(scores)} rows written to {DRIVE_ROOT}/edge_preservation_check.csv\n')

    report = pd.DataFrame({name: report_column(scores, name) for name in SCORE_COLUMNS})
    report.to_csv(f'{DRIVE_ROOT}/edge_preservation_report.csv')
    print(report.to_string())
