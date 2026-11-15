# Tests the transform x rule interaction with an aligned rank transform (ART) ANOVA (Wobbrock et al. 2011), 
# with patient as the repeated-measures factor.
# Run after 03. Reads metrics_long.csv only (nothing is fused again).

# ART: to test one effect, 
# subtract the other effects from every score (align), rank what is left, and run an ordinary ANOVA on the ranks.
# Each alignment is only valid for its own effect, so the ANOVA runs once per effect.

# Two grids:
#   5x3  all five transforms by all three rules
#   5x2  without the average rule, where four transforms give the same image

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from scipy.stats import rankdata
from statsmodels.stats.anova import AnovaRM

ALPHA = 0.05
TRANSFORMS = ['averaging', 'laplacian', 'dwt', 'swt', 'densefuse']
GRIDS = {'5x3': ['average', 'l1norm', 'maxabs'],
         '5x2': ['l1norm', 'maxabs']}
EFFECTS = ['transform', 'rule', 'transform:rule']

# MRI-PET has 3 patients (too few to test)
TESTED_PAIRINGS = ['mri_ct', 'mri_spect']

# One value per patient: average that patient's slices (same as 04)
def average_within_patients(table):
    rows = table[table['shift_px'] == 0]
    keys = ['pairing', 'case', 'transform', 'rule', 'metric']
    return rows.groupby(keys, as_index=False)['value'].mean()


# One row per patient and cell. Patients missing a cell are dropped
def build_long_table(patient_means, pairing, metric, rules):
    rows = patient_means[(patient_means['pairing'] == pairing)
                         & (patient_means['metric'] == metric)
                         & (patient_means['transform'].isin(TRANSFORMS))
                         & (patient_means['rule'].isin(rules))]
    cells_per_patient = rows.groupby('case')['value'].count()
    complete = cells_per_patient[cells_per_patient == len(TRANSFORMS) * len(rules)].index

    columns = ['case', 'transform', 'rule', 'value']
    return rows[rows['case'].isin(complete)][columns].reset_index(drop=True)

# Keep the residual plus the one effect being tested
def align(long_table, effect):
    values = long_table['value']
    cell_mean = long_table.groupby(['transform', 'rule'])['value'].transform('mean')
    transform_mean = long_table.groupby('transform')['value'].transform('mean')
    rule_mean = long_table.groupby('rule')['value'].transform('mean')
    grand_mean = values.mean()

    if effect == 'transform':
        effect_part = transform_mean - grand_mean
    elif effect == 'rule':
        effect_part = rule_mean - grand_mean
    else:
        effect_part = cell_mean - transform_mean - rule_mean + grand_mean
    return (values - cell_mean) + effect_part

# Two-way repeated-measures ANOVA, both factors within patient
def repeated_measures_anova(long_table, column):
    model = AnovaRM(long_table, depvar=column, subject='case',
                    within=['transform', 'rule'])
    return model.fit().anova_table

# Orthonormal contrasts for one factor: each column compares a level with the average of the levels before it
def contrasts_for(level_count):
    matrix = np.zeros((level_count, level_count - 1))
    for column in range(level_count - 1):
        matrix[:column + 1, column] = 1.0
        matrix[column + 1, column] = -(column + 1)
        matrix[:, column] /= np.linalg.norm(matrix[:, column])
    return matrix

# Greenhouse-Geisser epsilon: 1 if sphericity holds, smaller as it fails more.
# Both degrees of freedom are multiplied by it before p is called on
def greenhouse_geisser_epsilon(long_table, column, effect):
    wide = long_table.pivot(index='case', columns=['transform', 'rule'], values=column)
    wide = wide.sort_index(axis=1)      # cells ordered by transform, then rule

    transform_count = long_table['transform'].nunique()
    rule_count = long_table['rule'].nunique()
    transform_mean = np.full((transform_count, 1), 1.0 / np.sqrt(transform_count))
    rule_mean = np.full((rule_count, 1), 1.0 / np.sqrt(rule_count))

    if effect == 'transform':
        contrasts = np.kron(contrasts_for(transform_count), rule_mean)
    elif effect == 'rule':
        contrasts = np.kron(transform_mean, contrasts_for(rule_count))
    else:
        contrasts = np.kron(contrasts_for(transform_count), contrasts_for(rule_count))

    # contrast scores per patient, then their covariance across patients
    scores = wide.to_numpy() @ contrasts
    covariance = np.atleast_2d(np.cov(scores, rowvar=False))
    df_effect = contrasts.shape[1]
    epsilon = np.trace(covariance) ** 2 / (df_effect * np.sum(covariance ** 2))
    return float(min(epsilon, 1.0))

# Holm correction (same as from script 04)
def holm_adjust(pvalues):
    count = len(pvalues)
    adjusted = np.empty(count)
    running = 0.0
    for position, index in enumerate(np.argsort(pvalues)):
        running = max(running, (count - position) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted

# The three effects for one pairing, metric and grid
def test_one_metric(long_table, pairing, metric, grid):
    rows = []

    for effect in EFFECTS:
        table = long_table.copy()
        table['aligned'] = align(table, effect)
        table['rank'] = rankdata(table['aligned'])      # ties get their average rank

        result = repeated_measures_anova(table, 'rank').loc[effect]
        f_value = float(result['F Value'])
        df_effect = float(result['Num DF'])
        df_error = float(result['Den DF'])
        epsilon = greenhouse_geisser_epsilon(table, 'rank', effect)
        p_gg = float(f_distribution.sf(f_value, epsilon * df_effect, epsilon * df_error))

        # Alignment check: before ranking, the removed effects should have F near zero
        others = [name for name in EFFECTS if name != effect]
        leftover = repeated_measures_anova(table, 'aligned').loc[others, 'F Value'].max()

        rows.append({
            'pairing': pairing,
            'metric': metric,
            'grid': grid,
            'effect': effect,
            'patients': table['case'].nunique(),
            'f_value': f_value,
            'df_effect': df_effect,
            'df_error': df_error,
            'p_raw': float(result['Pr > F']),       # no sphericity correction
            'epsilon': epsilon,
            'p_gg': p_gg,                           # Greenhouse-Geisser corrected
            'p_holm': np.nan,                       # filled in by run()
            'partial_eta_sq': f_value * df_effect / (f_value * df_effect + df_error),
            'alignment_check': float(leftover),
        })
    return rows

def run(csv_path, output_path):
    patient_means = average_within_patients(pd.read_csv(csv_path))
    metrics = sorted(patient_means['metric'].unique())

    all_rows = []
    for pairing in TESTED_PAIRINGS:
        for grid, rules in GRIDS.items():
            for metric in metrics:
                long_table = build_long_table(patient_means, pairing, metric, rules)
                all_rows.extend(test_one_metric(long_table, pairing, metric, grid))

    results = pd.DataFrame(all_rows)

    # Holm on the corrected p-values, across the seven measures of each pairing, grid and effect
    for _, block in results.groupby(['pairing', 'grid', 'effect']):
        results.loc[block.index, 'p_holm'] = holm_adjust(block['p_gg'].to_numpy())
    results.to_csv(output_path, index=False)
    print(f'{len(results)} rows written to {output_path}')
    worst = results['alignment_check'].max()
    print(f'alignment check, largest leftover F {worst:.1e} -> {"ok" if worst < 1e-6 else "FAIL"}')
    return results

def format_p(value):
    return f'{value:.4f}' if value >= 1e-4 else f'{value:.1e}'


# One block per pairing and grid, the three effects side by side
def print_results(results):
    for (pairing, grid), block in results.groupby(['pairing', 'grid'], sort=False):
        patients = int(block['patients'].iloc[0])
        print(f'\n{pairing}, {grid} grid, {patients} patients   '
              f'(F, Holm p, partial eta squared)')
        print(f'  {"metric":<18} {"transform":>26} {"rule":>26} {"transform x rule":>26}')
        for metric, rows in block.groupby('metric', sort=False):
            line = f'  {metric:<18}'
            for effect in EFFECTS:
                row = rows[rows['effect'] == effect].iloc[0]
                mark = '*' if row['p_holm'] < ALPHA else ' '
                line += (f' {row["f_value"]:>8.1f} {format_p(row["p_holm"]):>9}{mark}'
                         f' {row["partial_eta_sq"]:>6.3f}')
            print(line)

if __name__ == '__main__':
    results = run(f'{DRIVE_ROOT}/metrics_long.csv', f'{DRIVE_ROOT}/art_anova.csv')
    print_results(results)
