# Follow-up to 09 for the camera-ready paper.
# Writes out three things the paper reports that no earlier notebook produces
# Run after 05 and 09. 

import importlib.util
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import metrics
from evaluation import crop, load_grayscale, pair_names
from pipeline import MODALITIES, MRI_TOKEN
from utils import load_image_pair

SHIFTS = [0, 1, 2, 4]
CROP_MARGIN = max(SHIFTS)    # same crop as 05
POWERS = [1.5, 1.0]          # the published power first, then the check


# these numbers come from the same code as 09's report
def load_notebook_09(path='notebooks/09_check_edge_preservation.py'):
    spec = importlib.util.spec_from_file_location('notebook_09', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The unfused MRI's score splits into the MRI's share of the edge weight
# and how well the MRI keeps the partner's edges (the partner term)
def score_breakdown(mri, partner, power):
    mri_strength, mri_orientation = metrics.sobel_edges(mri)
    partner_strength, partner_orientation = metrics.sobel_edges(partner)
    partner_kept = metrics.edges_kept(partner_strength, partner_orientation,
                                      mri_strength, mri_orientation)
    mri_weight = mri_strength ** power
    partner_weight = partner_strength ** power
    share = mri_weight.sum() / (mri_weight.sum() + partner_weight.sum())
    partner_term = (partner_kept * partner_weight).sum() / partner_weight.sum()
    return share, partner_term


# 05 names a fused folder like swt__maxabs, or swt__maxabs__shift2 once shifted
def shift_folder(shift_root, transform, shift):
    name = f'{transform}__maxabs'
    if shift:
        name = f'{name}__shift{shift}'
    return f'{shift_root}/{name}/fused_only'


# One row per image pair for the breakdown, one row per fused image for the shift test
def score_everything(data_root, shift_root):
    breakdown_rows = []
    shift_rows = []

    for subfolder, partner_token, modality in MODALITIES:
        folder = f'{data_root}/{subfolder}'
        mri_files = sorted(f for f in os.listdir(folder) if MRI_TOKEN in f and f.endswith('.png'))
        for mri_file in mri_files:
            partner_file = mri_file.replace(MRI_TOKEN, partner_token)
            mri, partner = load_image_pair(f'{folder}/{mri_file}', f'{folder}/{partner_file}')
            case, slice_number, fused_name = pair_names(mri_file, modality.lower())

            # Table V uses the whole image, like the main results
            for power in POWERS:
                share, partner_term = score_breakdown(mri, partner, power)
                breakdown_rows.append({'pairing': subfolder, 'case': case, 'slice': slice_number,
                                       'power': power, 'share': share, 'partner_term': partner_term})

            # the shift test crops every image, like 05
            mri_cropped = crop(mri, CROP_MARGIN)
            partner_cropped = crop(partner, CROP_MARGIN)
            for transform in ['dwt', 'swt']:
                for shift in SHIFTS:
                    fused = load_grayscale(f'{shift_folder(shift_root, transform, shift)}/{fused_name}')
                    fused = crop(fused, CROP_MARGIN)
                    for power in POWERS:
                        metrics.EDGE_WEIGHT_POWER = power
                        score = metrics.edge_preservation(mri_cropped, partner_cropped, fused)
                        shift_rows.append({'pairing': subfolder, 'case': case, 'slice': slice_number,
                                           'transform': transform, 'shift_px': shift,
                                           'power': power, 'value': score})
        print(f'  {subfolder}: {len(mri_files)} pairs')

    metrics.EDGE_WEIGHT_POWER = 1.5    # back to the published setting
    return pd.DataFrame(breakdown_rows), pd.DataFrame(shift_rows)


# Slices averaged within a patient, then patients averaged
def patient_means(rows, keys, columns):
    per_patient = rows.groupby(keys + ['case'])[columns].mean()
    return per_patient.groupby(keys).mean()


# The SWT minus the DWT at each shift, one row per power and pairing
def shift_gaps(shift_rows):
    means = patient_means(shift_rows, ['power', 'pairing', 'transform', 'shift_px'], 'value')
    table = means.unstack('transform')
    return (table['swt'] - table['dwt']).unstack('shift_px')


# Section III-F's two numbers for the CDDFuse code, from 09's reference scores
def cddfuse_numbers(check_csv, notebook_09):
    scores = pd.read_csv(check_csv, dtype={'slice': str})
    spect = scores[scores['pairing'] == 'mri_spect']

    # the dominance test inside MRI-SPECT, as in 09's report
    means = notebook_09.cell_means(spect, 'reference')
    wide = means.drop('unfused', level='transform').unstack('rule')
    rule_spread = (wide.max(axis=1) - wide.min(axis=1)).min()
    transform_spread = (wide.max(axis=0) - wide.min(axis=0)).max()

    # the planned family on MRI-SPECT, as in 09's rule_comparisons_won
    patients = (spect.groupby(['case', 'transform', 'rule'])['reference'].mean()
                     .unstack(['transform', 'rule']).dropna())
    differences = [(patients[(first, first_rule)] - patients[(second, second_rule)]).to_numpy()
                   for first, first_rule, second, second_rule in notebook_09.PLANNED]
    pvalues = np.array([1.0 if np.all(d == 0) else float(wilcoxon(d).pvalue) for d in differences])
    adjusted = notebook_09.holm_adjust(pvalues)
    dwt = notebook_09.PLANNED.index(('dwt', 'l1norm', 'dwt', 'maxabs'))
    return rule_spread - transform_spread, float(np.median(differences[dwt])), float(adjusted[dwt])


if __name__ == '__main__':
    notebook_09 = load_notebook_09()
    breakdown, shifts = score_everything(DATA, f'{DRIVE_ROOT}/results_misalignment')

    # 1. Table V
    table_v = patient_means(breakdown, ['power', 'pairing'], ['share', 'partner_term'])
    table_v.to_csv(f'{DRIVE_ROOT}/unfused_breakdown.csv')
    print('\nTable V breakdown (the paper prints the power 1.5 rows)')
    print(table_v.round(4).to_string())

    # 2. the shift test at both powers. Power 1.5 has to match 05's committed scores first
    shifts.to_csv(f'{DRIVE_ROOT}/misalignment_edge_by_power.csv', index=False)
    gaps = shift_gaps(shifts)
    committed = pd.read_csv('results/metrics_misalignment.csv')
    committed = committed[(committed['metric'] == 'edge_preservation') &
                          committed['transform'].isin(['dwt', 'swt'])].copy()
    committed['power'] = 1.5
    largest = (gaps.loc[1.5] - shift_gaps(committed).loc[1.5]).abs().max().max()
    print(f'\npower 1.5 against results/metrics_misalignment.csv: largest difference {largest:.1e}')
    for power in POWERS:
        gap = gaps.loc[power]
        print(f'\npower {power}: SWT minus DWT, mean edge preservation')
        print(gap.round(4).to_string())
        print('  SWT above DWT at every shift in every pairing:', bool((gap > 0).all().all()))
        print('  gap narrower at 4 px than at 0 px in every pairing:', bool((gap[4] < gap[0]).all()))

    # 3. the CDDFuse code's own numbers
    margin, median, p_value = cddfuse_numbers('results/edge_preservation_check.csv', notebook_09)
    print(f'\nCDDFuse code, MRI-SPECT: smallest rule spread minus largest transform spread {margin:+.4f}')
    print(f'CDDFuse code, MRI-SPECT, DWT l1norm against maxabs: median difference {median:+.4f}, '
          f'Holm p {p_value:.4f}')
