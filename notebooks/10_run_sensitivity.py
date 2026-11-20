# Sensitivity check: do the main results hold under a different setting?
# Reruns MRI-CT with one setting changed at a time:
#   levels3 - DWT and SWT at 3 levels instead of 2
#   window5 - l1norm with a 5x5 window instead of 3x3
# then compares each rerun with the main grid.

import importlib.util
import os
import numpy as np
import pandas as pd
import metrics
from coeffs import make_fusion_function
from evaluation import score_everything
from pipeline import fuse_all_pairs
from rules import RULES, l1norm
from transforms import TRANSFORMS, dwt, swt

PAIRING = 'mri_ct'
TRANSFORM_NAMES = ['averaging', 'laplacian', 'dwt', 'swt', 'densefuse']
RULE_NAMES = ['average', 'l1norm', 'maxabs']
QUESTIONS = ['dominant', 'l1_higher', 'swt_above_dwt', 'unfused_above']

# These three cells repeat averaging__average, so twelve of the fifteen are distinct (same as 04)
DUPLICATE_CELLS = [('laplacian', 'average'), ('dwt', 'average'), ('swt', 'average')]

# A wavelet transform at another depth. The inverse does not need the depth
def at_depth(module, levels):
    def forward(image):
        return module.forward(image, levels=levels)
    return (forward, module.inverse)
def l1norm_with_window(window_size):
    def rule(band_a, band_b):
        return l1norm(band_a, band_b, window_size=window_size)
    return rule

# name -> (transforms, rules), each with one setting changed
def build_variants():
    deeper = dict(TRANSFORMS)
    deeper['dwt'] = at_depth(dwt, 3)
    deeper['swt'] = at_depth(swt, 3)
    wider = dict(RULES)
    wider['l1norm'] = l1norm_with_window(5)
    return {'levels3': (deeper, RULES),
            'window5': (TRANSFORMS, wider)}

# fuse_all_pairs fuses every pairing folder it finds, so link the one pairing into a folder of its own
def single_pairing_root(data_root, pairing, link_root='/content/sensitivity_data'):
    os.makedirs(link_root, exist_ok=True)
    link = f'{link_root}/{pairing}'
    if not os.path.exists(link):
        os.symlink(f'{data_root}/{pairing}', link)
    return link_root

# Fuse and score all fifteen cells for one variant
def run_variant(transforms, rules, data_root, results_root, output_path):
    for transform_name in TRANSFORM_NAMES:
        for rule_name in RULE_NAMES:
            fusion_function = make_fusion_function(transforms[transform_name], rules[rule_name])
            fuse_all_pairs(fusion_function, f'{transform_name}__{rule_name}',
                           data_root, results_root)
    score_everything(data_root, results_root, output_path)

# One value per patient, metric and cell: that patient's slices averaged (same as 04).
# The unfused sources are included as transform 'unfused'
def patient_means(csv_path, pairing):
    table = pd.read_csv(csv_path)
    rows = table[(table['pairing'] == pairing) & (table['shift_px'] == 0)]
    keys = ['metric', 'case', 'transform', 'rule']
    return rows.groupby(keys, as_index=False)['value'].mean()

# True if the first cell scores higher: median over patients of the paired difference (as in 04)
def scores_higher(by_patient, first, second):
    return (by_patient[first] - by_patient[second]).median() > 0

# One row of answers per metric:
#   dominant - factor that dominates by the spread test of 06
#   l1_higher - transforms where l1norm scores above maxabs
#   swt_above_dwt - selection rules under which the SWT scores above the DWT
#   unfused_above - distinct configurations the unfused MRI scores above
def answer_questions(per_patient):
    rows = []

    for metric, block in per_patient.groupby('metric'):
        # patients as rows, one column per (transform, rule) cell
        by_patient = block.pivot(index='case', columns=['transform', 'rule'], values='value')
        means = by_patient.mean()
        fused = means.drop('unfused', level='transform')
        wide = fused.unstack('rule')
        rule_spread = wide.max(axis=1) - wide.min(axis=1)
        transform_spread = wide.max(axis=0) - wide.min(axis=0)

        if rule_spread.min() > transform_spread.max():
            dominant = 'rule'
        elif transform_spread.min() > rule_spread.max():
            dominant = 'transform'
        else:
            dominant = 'neither'

        l1_higher = sum(scores_higher(by_patient, (name, 'l1norm'), (name, 'maxabs'))
                        for name in wide.index)
        swt_above_dwt = sum(scores_higher(by_patient, ('swt', rule), ('dwt', rule))
                            for rule in ('l1norm', 'maxabs'))
        distinct = fused.drop(DUPLICATE_CELLS, errors='ignore')
        unfused_above = (distinct < means[('unfused', 'mri')]).sum()
        rows.append({'metric': metric,
                     'dominant': dominant,
                     'l1_higher': f'{int(l1_higher)} of {len(wide)}',
                     'swt_above_dwt': f'{int(swt_above_dwt)} of 2',
                     'unfused_above': f'{int(unfused_above)} of {len(distinct)}'})
    return pd.DataFrame(rows)

# One row per metric and question, main grid next to the variant
def compare(baseline, variant, name):
    merged = baseline.merge(variant, on='metric', suffixes=('_main', '_variant'))
    rows = []
    for _, row in merged.iterrows():
        for question in QUESTIONS:
            before = row[f'{question}_main']
            after = row[f'{question}_variant']
            rows.append({'variant': name, 'metric': row['metric'], 'question': question,
                         'main_grid': before, 'variant_value': after,
                         'same': before == after})
    return pd.DataFrame(rows)

# 04's test code, so these p-values come from the same code as the paper's
def load_notebook_04(path='notebooks/04_run_statistics.py'):
    spec = importlib.util.spec_from_file_location('notebook_04', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 04's planned family on one pairing: a Wilcoxon test over patients for each comparison,
# Holm corrected within each measure's seven
def planned_tests(csv_path, notebook_04, label):
    patients = notebook_04.average_within_patients(pd.read_csv(csv_path))
    rows = []
    for metric in sorted(patients['metric'].unique()):
        wide = notebook_04.build_matrix(patients, PAIRING, metric)
        rows += notebook_04.test_family(wide, PAIRING, metric, notebook_04.PLANNED_COMPARISONS, 'planned')
    tests = pd.DataFrame(rows)
    tests.insert(0, 'variant', label)
    return tests


if __name__ == '__main__':
    # metrics_long.csv was scored at the published edge weight power, so the reruns have to as well
    metrics.EDGE_WEIGHT_POWER = 1.5
    data_root = single_pairing_root(DATA, PAIRING)
    baseline = answer_questions(patient_means(f'{DRIVE_ROOT}/metrics_long.csv', PAIRING))
    variants = build_variants()
    comparisons = []
    for name, (transforms, rules) in variants.items():
        print(f'\n{name}')
        results_root = f'{DRIVE_ROOT}/results_sensitivity/{name}'
        output_path = f'{DRIVE_ROOT}/metrics_sensitivity_{name}.csv'
        run_variant(transforms, rules, data_root, results_root, output_path)

        variant = answer_questions(patient_means(output_path, PAIRING))
        comparisons.append(compare(baseline, variant, name))
    summary = pd.concat(comparisons, ignore_index=True)
    summary.to_csv(f'{DRIVE_ROOT}/sensitivity_summary.csv', index=False)
    print(f'\nmain grid, {PAIRING}')
    print(baseline.to_string(index=False))

    for name, block in summary.groupby('variant', sort=False):
        changed = block[~block['same']]
        print(f'\n{name}: {len(block) - len(changed)} of {len(block)} answers unchanged')
        if not changed.empty:
            columns = ['metric', 'question', 'main_grid', 'variant_value']
            print(changed[columns].to_string(index=False))

    # 04's planned tests on the main grid and on each rerun
    notebook_04 = load_notebook_04()
    tests = [planned_tests(f'{DRIVE_ROOT}/metrics_long.csv', notebook_04, 'main_grid')]
    for name in variants:
        tests.append(planned_tests(f'{DRIVE_ROOT}/metrics_sensitivity_{name}.csv', notebook_04, name))
    tests = pd.concat(tests, ignore_index=True)
    tests.to_csv(f'{DRIVE_ROOT}/sensitivity_tests.csv', index=False)

    # every planned comparison whose median changed sign, with the rerun's corrected p
    keys = ['metric', 'comparison']
    main_grid = tests[tests['variant'] == 'main_grid'].set_index(keys)
    for name in variants:
        rerun = tests[tests['variant'] == name].set_index(keys)
        before = main_grid['statistic'].reindex(rerun.index)
        flipped = rerun[np.sign(rerun['statistic']) != np.sign(before)]
        print(f'\n{name}: {len(flipped)} of {len(rerun)} planned comparisons changed direction')
        for (metric, comparison), row in flipped.iterrows():
            print(f'  {metric}, {comparison}: median {before[(metric, comparison)]:+.4f} -> '
                  f'{row["statistic"]:+.4f}, corrected p {row["p_holm"]:.4f}')
