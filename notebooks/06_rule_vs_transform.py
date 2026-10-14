# Builds the three tables the results section needs:
# the mean for every cell, which rule scores higher on each measure, and whether the rule or the transform moves results further

# Run after 04. Reads metrics_long.csv for the means and statistics_tests.csv for the orderings. 
import re
import pandas as pd

TRANSFORMS = ['averaging', 'laplacian', 'dwt', 'swt', 'densefuse']
RULES = ['average', 'l1norm', 'maxabs']

# The identity row applies the rule to raw pixels, which is a pixel-wise max rather than coefficient selection, 
# so it sits at the extreme on most measures.
FIXED_BASES = ['laplacian', 'dwt', 'swt']

# Measures that compare the fused image against the sources.
# The other three are computed on the fused image alone. 
# This split is for the SWT against DWT result and is not the same split as the rule axis, which is why both are printed.
SOURCE_REFERENCED = ['MI', 'SSIM_mri', 'SSIM_partner', 'edge_preservation']

# matches the planned comparisons that hold the transform fixed and vary the rule
WITHIN_TRANSFORM = re.compile(r'(\w+)__l1norm vs \1__maxabs$')
ALPHA = 0.05

# Average each patient's slices, then average over patients, so one patient counts once
def cell_means(csv_path):
    table = pd.read_csv(csv_path)
    fused = table[table['transform'] != 'unfused']
    per_patient = (fused.groupby(['metric', 'pairing', 'case', 'transform', 'rule'],
                                 as_index=False)['value'].mean())
    return per_patient.groupby(['metric', 'transform', 'rule'], as_index=False)['value'].mean()

# Vary one factor with the other held fixed, 
# then compare the smallest rule effect against the largest transform effect
def dominant_factor(block, transforms=TRANSFORMS):
    scoped = block[block['transform'].isin(transforms)]
    rule = scoped.groupby('transform')['value'].agg(lambda v: v.max() - v.min()).min()
    transform = scoped.groupby('rule')['value'].agg(lambda v: v.max() - v.min()).max()
    return rule, transform, 'rule' if rule > transform else 'transform'


# Which rule scores higher on each measure. 
# Read from 04's output rather than recomputed
def rule_orderings(tests_path):
    tests = pd.read_csv(tests_path)
    within = tests[tests['comparison'].apply(lambda c: bool(WITHIN_TRANSFORM.match(c)))]
    rows = []
    for metric, block in within.groupby('metric'):
        rows.append({'metric': metric,
                     'uses_sources': 'yes' if metric in SOURCE_REFERENCED else 'no',
                     'l1': (block['statistic'] > 0).sum(),
                     'maxabs': (block['statistic'] < 0).sum(),
                     'significant': (block['p_holm'] < ALPHA).sum(),
                     'comparisons': len(block)})
    return pd.DataFrame(rows)


def factor_verdicts(means):
    rows = []
    for metric, block in means.groupby('metric'):
        rule, transform, who = dominant_factor(block)
        rows.append({'metric': metric,
                     'rule': rule,
                     'transform': transform,
                     'ratio': rule / transform,
                     'dominant': who,
                     'narrow': dominant_factor(block, FIXED_BASES)[2]})
    return pd.DataFrame(rows)


def print_cells(means):
    print('\nmean by cell, patient level')
    for metric, block in means.groupby('metric'):
        wide = block.pivot(index='transform', columns='rule', values='value')
        print(f'\n  {metric}')
        print(' ' * 16 + ''.join(f'{r:>12}' for r in RULES))
        for name in TRANSFORMS:
            print(f'    {name:<12}' + ''.join(f'{wide.loc[name, r]:>12.4f}' for r in RULES))


def print_orderings(table):
    print('\n\nwhich rule scores higher, l1 against max-magnitude')
    print(f'  {"measure":<20}{"uses sources":>14}{"l1":>6}{"max-mag":>9}{"significant":>13}')
    for row in table.itertuples():
        print(f'  {row.metric:<20}{row.uses_sources:>14}{row.l1:>6}{row.maxabs:>9}'
              f'{row.significant:>9}/{row.comparisons:<3}')


def print_verdicts(table):
    print('\n\nwhich factor moves results further')
    print(f'  {"measure":<20}{"rule":>9}{"transform":>11}{"ratio":>8}'
          f'{"dominant":>11}{"3 bases only":>15}')
    for row in table.itertuples():
        print(f'  {row.metric:<20}{row.rule:>9.4f}{row.transform:>11.4f}{row.ratio:>8.2f}'
              f'{row.dominant:>11}{row.narrow:>15}')

    wins = (table['dominant'] == 'rule').sum()
    print(f'\n  the rule dominates on {wins} of {len(table)} measures')


if __name__ == '__main__':
    means = cell_means(f'{DRIVE_ROOT}/metrics_long.csv')
    orderings = rule_orderings(f'{DRIVE_ROOT}/statistics_tests.csv')
    verdicts = factor_verdicts(means)

    means.to_csv(f'{DRIVE_ROOT}/cell_means.csv', index=False)
    orderings.to_csv(f'{DRIVE_ROOT}/rule_orderings.csv', index=False)
    verdicts.to_csv(f'{DRIVE_ROOT}/factor_verdicts.csv', index=False)

    print_cells(means)
    print_orderings(orderings)
    print_verdicts(verdicts)
