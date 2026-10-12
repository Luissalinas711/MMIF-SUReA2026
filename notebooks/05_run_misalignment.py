# Tests whether the SWT lead over the DWT comes from shift invariance.
# Shifts the partner by 0, 1, 2 and 4 pixels before fusing and measures how fast each transform degrades 
# If shift invariance is the reason, the DWT should fall off faster and the gap widen

# Run after 04_run_statistics.py!!! 
# maxabs is held: conventional, and where the SWT lead is largest.

import pandas as pd

from evaluation import score_everything
from pipeline import run_all_combinations

SHIFTS = [0, 1, 2, 4]
HELD_RULE = 'maxabs'

# Shifting leaves a black band on the top and left that grows with the shift. 
# Crop every image by the largest one, shift 0 included, so the numbers compare fusion quality (not border size)
CROP_MARGIN = max(SHIFTS)


def run_shifts(data_root, results_root):
    for shift in SHIFTS:
        print(f'\nshift {shift} px')
        run_all_combinations(data_root, results_root,
                             rule_names=[HELD_RULE],
                             shift_pixels=shift)


# Mean score at each shift, one row per transform, so the degradation reads as a slope.
# The CSV holds one rule by construction, so there is no rule to filter.
# The unfused references are dropped
def degradation_table(csv_path, metric='edge_preservation'):
    table = pd.read_csv(csv_path)
    rows = table[(table['metric'] == metric) & (table['transform'] != 'unfused')]

    # average each patient's slices first, same as the main statistics
    patient_means = (rows.groupby(['pairing', 'case', 'transform', 'shift_px'], as_index=False)
                         ['value'].mean())

    wide = patient_means.pivot_table(index=['pairing', 'transform'],
                                     columns='shift_px', values='value')

    # how much is lost going from aligned to the largest shift, absolute and as a share
    # the share matters b/c transforms start at different heights, so the absolute drop flatters whichever one had least to lose
    wide['drop'] = wide[SHIFTS[0]] - wide[SHIFTS[-1]]
    wide['drop_pct'] = wide['drop'] / wide[SHIFTS[0]]

    return wide.round(4)


if __name__ == '__main__':
    RESULTS = f'{DRIVE_ROOT}/results_misalignment'
    OUTPUT = f'{DRIVE_ROOT}/metrics_misalignment.csv'

    run_shifts(DATA, RESULTS)
    score_everything(DATA, RESULTS, OUTPUT, crop_margin=CROP_MARGIN)

    print(f'\nmean edge preservation by shift, {HELD_RULE} rule')
    print(degradation_table(OUTPUT))
