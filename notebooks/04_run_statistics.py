# Tests the grid and writes the numbers the results section needs.
# Patient level, so one patient counts once however many slices they contributed.
# Writes two files: the descriptive means (including the unfused references) and the test results
# Two comparison families, corrected separately

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ALPHA = 0.05

# The four linear transforms give the same image under the average rule
# so three of them are duplicates and only averaging__average is kept
# this means we have 15 cells, 12 distinct configurations
CONFIGURATIONS = [
    'averaging__average', 'averaging__l1norm', 'averaging__maxabs', 'laplacian__l1norm',
    'laplacian__maxabs', 'dwt__l1norm', 'dwt__maxabs', 'swt__l1norm',
    'swt__maxabs', 'densefuse__average', 'densefuse__l1norm', 'densefuse__maxabs',
]

# Fixed before looking at the numbers.
# Two comparisons ask whether dropping decimation helps
# five ask whether soft selection differs from hard selection
# Keeping the list short is what makes MRI-CT testable at all.
# With 10 patients the smallest two-sided Wilcoxon p is 0.00195, and Holm's strictest threshold over 7 tests is 0.05/7 = 0.00714.
# Over all 66 pairs it would be 0.00076, which no MRI-CT comparison could clear
PLANNED_COMPARISONS = [
    ('swt__l1norm', 'dwt__l1norm'), ('swt__maxabs', 'dwt__maxabs'),
    ('averaging__l1norm', 'averaging__maxabs'), ('laplacian__l1norm', 'laplacian__maxabs'),
    ('dwt__l1norm', 'dwt__maxabs'), ('swt__l1norm', 'swt__maxabs'),
    ('densefuse__l1norm', 'densefuse__maxabs'),
]

# Added after the planned set and corrected on its own
# Completes all three pairwise rule contrasts in every transform
# Eight of the ten start from averaging__average,
# because the linear transforms collapse to one image under the average rule, so only DenseFuse has a baseline of its own.
# Ten still fits. At 10 patients the floor is 0.00195 against 0.05/10 = 0.005
POSTHOC_COMPARISONS = [
    ('averaging__average', 'averaging__l1norm'),
    ('averaging__average', 'averaging__maxabs'),
    ('averaging__average', 'laplacian__l1norm'),
    ('averaging__average', 'laplacian__maxabs'),
    ('averaging__average', 'dwt__l1norm'),
    ('averaging__average', 'dwt__maxabs'),
    ('averaging__average', 'swt__l1norm'),
    ('averaging__average', 'swt__maxabs'),
    ('densefuse__average', 'densefuse__l1norm'),
    ('densefuse__average', 'densefuse__maxabs'),
]

# Below this a comparison cannot clear the correction however large the effect,
# since the smallest two-sided Wilcoxon p at n patients is 2^(1-n).
# n = 9 gives 0.0039, which clears 0.05/7 and also the stricter 0.05/10 of the post hoc set.
# MRI-PET has 3 patients, where the floor is 0.25, so it is described and not tested
MIN_PATIENTS_TO_TEST = 9

# Average each patient's slices so one patient counts once
def average_within_patients(table, shift_px=0):
    rows = table[table['shift_px'] == shift_px]
    return (rows.groupby(['pairing', 'case', 'transform', 'rule', 'metric'], as_index=False)
                ['value'].mean())


def configuration_name(frame):
    return frame['transform'] + '__' + frame['rule']


# Patients down the rows, configurations across the columns.
# Patients missing any configuration are dropped, b/c Friedman and Wilcoxon both need complete rows
def build_matrix(patient_means, pairing, metric):
    subset = patient_means[(patient_means['pairing'] == pairing)
                           & (patient_means['metric'] == metric)].copy()
    subset['configuration'] = configuration_name(subset)

    wide = subset.pivot(index='case', columns='configuration', values='value')

    present = [name for name in CONFIGURATIONS if name in wide.columns]
    return wide[present].dropna()


# The effect size that goes with Friedman
# 0 means every patient ranks the configurations differently,
# 1 means they all rank them the same way
def kendalls_w(chi_square, patients, configurations):
    return float(chi_square / (patients * (configurations - 1)))


# The effect size that goes with Wilcoxon.
# the sign says which configuration won and the magnitude says how consistently it wins across patients
def rank_biserial(differences):
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0

    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()

    return float((positive - negative) / (positive + negative))


# Holm, applied within each metric, pairing and family.
# Multiply the smallest p by the number of tests, the next by one less, and so on
# then force the sequence to never decrease
def holm_adjust(pvalues):
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count)
    running = 0.0

    for position, index in enumerate(order):
        running = max(running, (count - position) * pvalues[index])
        adjusted[index] = min(running, 1.0)

    return adjusted


# One family of Wilcoxon comparisons, corrected against itself only.
# Keeping the families apart is what lets the planned set stay confirmatory.
def test_family(wide, pairing, metric, comparisons, family):
    rows = []
    for first, second in comparisons:
        if first not in wide.columns or second not in wide.columns:
            continue

        differences = wide[first].to_numpy() - wide[second].to_numpy()

        if np.all(differences == 0):
            pvalue = 1.0
        else:
            pvalue = float(wilcoxon(differences).pvalue)

        rows.append({
            'pairing': pairing,
            'metric': metric,
            'test': 'wilcoxon',
            'family': family,
            'comparison': f'{first} vs {second}',
            'patients': len(wide),
            'statistic': float(np.median(differences)),
            'p_raw': pvalue,
            'p_holm': np.nan,
            'effect_size': rank_biserial(differences),
            'effect_name': 'rank_biserial',
        })

    adjusted = holm_adjust(np.array([row['p_raw'] for row in rows]))
    for row, value in zip(rows, adjusted):
        row['p_holm'] = float(value)

    return rows


# Friedman across the 12 configurations, then each comparison family in turn
def test_one_metric(wide, pairing, metric):
    patients = len(wide)
    columns = [wide[name].to_numpy() for name in wide.columns]

    chi_square, friedman_p = friedmanchisquare(*columns)

    friedman = {
        'pairing': pairing,
        'metric': metric,
        'test': 'friedman',
        'family': 'omnibus',
        'comparison': f'{len(wide.columns)} configurations',
        'patients': patients,
        'statistic': float(chi_square),
        'p_raw': float(friedman_p),
        'p_holm': np.nan,              # Friedman is one test, nothing to correct against
        'effect_size': kendalls_w(chi_square, patients, len(wide.columns)),
        'effect_name': 'kendalls_w',
    }

    return ([friedman]
            + test_family(wide, pairing, metric, PLANNED_COMPARISONS, 'planned')
            + test_family(wide, pairing, metric, POSTHOC_COMPARISONS, 'post hoc'))


# Means and standard deviations at the patient level (unfused references included)
def describe(patient_means):
    summary = patient_means.copy()
    summary['configuration'] = configuration_name(summary)

    return (summary.groupby(['pairing', 'metric', 'configuration'], as_index=False)['value']
                   .agg(['mean', 'std', 'count'])
                   .rename(columns={'count': 'patients'}))


def run(csv_path, summary_path, tests_path):
    table = pd.read_csv(csv_path)
    patient_means = average_within_patients(table)

    summary = describe(patient_means)
    summary.to_csv(summary_path, index=False)
    print(f'descriptive summary written to {summary_path}')

    metrics = sorted(patient_means['metric'].unique())
    all_results = []

    for pairing in ['mri_ct', 'mri_spect', 'mri_pet']:
        rows = patient_means[patient_means['pairing'] == pairing]
        if rows.empty:
            continue

        patients = rows['case'].nunique()
        print(f'\n{pairing}: {patients} patients')

        if patients < MIN_PATIENTS_TO_TEST:
            print(f'  under {MIN_PATIENTS_TO_TEST} patients, so no test here can reach '
                  f'significance. Descriptive only.')
            continue

        for metric in metrics:
            wide = build_matrix(patient_means, pairing, metric)

            if len(wide) < MIN_PATIENTS_TO_TEST or len(wide.columns) < 3:
                print(f'  [skip] {metric}: {len(wide)} complete patients, '
                      f'{len(wide.columns)} configurations')
                continue

            all_results.extend(test_one_metric(wide, pairing, metric))

    tests = pd.DataFrame(all_results)
    tests.to_csv(tests_path, index=False)
    print(f'\n{len(tests)} test results written to {tests_path}')

    return summary, tests


# Holm values at the Wilcoxon floor are far below 0.0001, so fixed notation prints them as 0.0000.
# Fixed where it is readable, scientific where it is not.
def format_p(value):
    return f'{value:.4f}' if value >= 1e-4 else f'{value:.1e}'


# One metric at a time, so any errors are visible and the source of error is clear.
# The families print separately b/c each one carries its own correction.
def print_tests(tests, metric='edge_preservation'):
    for pairing in tests['pairing'].unique():
        rows = tests[(tests['pairing'] == pairing) & (tests['metric'] == metric)]
        if rows.empty:
            continue

        friedman = rows[rows['test'] == 'friedman'].iloc[0]
        print(f'\n{pairing}, {metric}, {int(friedman["patients"])} patients')
        print(f'  Friedman chi2 = {friedman["statistic"]:.2f}, '
              f'p = {friedman["p_raw"]:.2e}, Kendall W = {friedman["effect_size"]:.3f}')

        for family in ['planned', 'post hoc']:
            block = rows[rows['family'] == family]
            if block.empty:
                continue

            print(f'\n  {family}, Holm over {len(block)}')
            print(f'  {"comparison":<42} {"median diff":>12} {"p (Holm)":>10} {"effect":>8}')

            for _, row in block.iterrows():
                mark = ' *' if row['p_holm'] < ALPHA else '  '
                print(f'    {row["comparison"]:<40} {row["statistic"]:>12.4f} '
                      f'{format_p(row["p_holm"]):>10} {row["effect_size"]:>8.3f}{mark}')


if __name__ == '__main__':
    summary, tests = run(f'{DRIVE_ROOT}/metrics_long.csv',
                         f'{DRIVE_ROOT}/statistics_summary.csv',
                         f'{DRIVE_ROOT}/statistics_tests.csv')

    for metric in tests['metric'].unique():
        print_tests(tests, metric)
