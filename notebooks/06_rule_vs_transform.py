# Builds the tables the results section needs:
# the mean for every cell, which rule scores higher on each measure, whether either factor dominates,
# and where the winning rule flips between transforms inside one pairing

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


# Vary one factor with the other held fixed. Returns the rule spreads and the transform spreads.
def spreads(block, transforms):
    scoped = block[block['transform'].isin(transforms)]
    return (scoped.groupby('transform')['value'].agg(lambda v: v.max() - v.min()),
            scoped.groupby('rule')['value'].agg(lambda v: v.max() - v.min()))


# One factor dominates only when its smallest spread beats the other's largest, which is what makes
# the claim hold in every cell rather than on average.
# Both tests have to be run. Neither passing is its own answer, not a win for the factor we did not test.
def verdict(rule, transform):
    if rule.min() > transform.max():
        return 'rule'
    if transform.min() > rule.max():
        return 'transform'
    return 'neither'


# Read once here and pass the frame on, since both of the ordering tables use the same rows
def within_transform(tests_path):
    tests = pd.read_csv(tests_path)
    return tests[tests['comparison'].apply(lambda c: bool(WITHIN_TRANSFORM.match(c)))]


# Which rule scores higher on each measure.
# Read from 04's output rather than recomputed
def rule_orderings(within):
    rows = []
    for metric, block in within.groupby('metric'):
        rows.append({'metric': metric,
                     'uses_sources': 'yes' if metric in SOURCE_REFERENCED else 'no',
                     'l1': (block['statistic'] > 0).sum(),
                     'maxabs': (block['statistic'] < 0).sum(),
                     'significant': (block['p_holm'] < ALPHA).sum(),
                     'comparisons': len(block)})
    return pd.DataFrame(rows)


# A ranking fails to carry when the winning rule flips between transforms inside one pairing, so this
# groups by both rather than pooling them.
# Confirmed means at least one comparison in each direction also clears Holm.
def rule_reversals(within):
    rows = []
    for (metric, pairing), block in within.groupby(['metric', 'pairing']):
        l1 = block[block['statistic'] > 0]
        maxabs = block[block['statistic'] < 0]
        flips = len(l1) > 0 and len(maxabs) > 0
        rows.append({'metric': metric,
                     'pairing': pairing,
                     'l1': len(l1),
                     'maxabs': len(maxabs),
                     'reversal': flips,
                     'confirmed': flips and (l1['p_holm'] < ALPHA).any()
                                        and (maxabs['p_holm'] < ALPHA).any()})
    return pd.DataFrame(rows)


# Both ends of each range, so we can check the verdict from the table ourself
def factor_verdicts(means):
    rows = []
    for metric, block in means.groupby('metric'):
        rule, transform = spreads(block, TRANSFORMS)
        rows.append({'metric': metric,
                     'rule_min': rule.min(),
                     'rule_max': rule.max(),
                     'transform_min': transform.min(),
                     'transform_max': transform.max(),
                     'dominant': verdict(rule, transform),
                     'narrow': verdict(*spreads(block, FIXED_BASES))})
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
    print(f'  {"measure":<20}{"rule spread":>20}{"transform spread":>22}'
          f'{"dominant":>11}{"3 bases only":>15}')
    for row in table.itertuples():
        rule = f'{row.rule_min:.4f} to {row.rule_max:.4f}'
        transform = f'{row.transform_min:.4f} to {row.transform_max:.4f}'
        print(f'  {row.metric:<20}{rule:>20}{transform:>22}{row.dominant:>11}{row.narrow:>15}')
    counts = table['dominant'].value_counts()
    print()
    for name in ('rule', 'transform', 'neither'):
        print(f'  {name} dominates on {counts.get(name, 0)} of {len(table)} measures')


def print_reversals(table):
    print('\n\nwhere the winning rule flips between transforms, within one pairing')
    print(f'  {"measure":<20}{"pairing":<12}{"l1":>5}{"max-mag":>9}   verdict')
    for row in table.itertuples():
        if row.confirmed:
            note = 'confirmed, both directions significant'
        elif row.reversal:
            note = 'flips, but one direction is not significant'
        else:
            note = 'no flip'
        print(f'  {row.metric:<20}{row.pairing:<12}{row.l1:>5}{row.maxabs:>9}   {note}')
    confirmed = table[table['confirmed']]['metric'].nunique()
    print(f'\n  confirmed on {confirmed} of {table["metric"].nunique()} measures')


if __name__ == '__main__':
    means = cell_means(f'{DRIVE_ROOT}/metrics_long.csv')
    within = within_transform(f'{DRIVE_ROOT}/statistics_tests.csv')

    orderings = rule_orderings(within)
    reversals = rule_reversals(within)
    verdicts = factor_verdicts(means)

    means.to_csv(f'{DRIVE_ROOT}/cell_means.csv', index=False)
    orderings.to_csv(f'{DRIVE_ROOT}/rule_orderings.csv', index=False)
    reversals.to_csv(f'{DRIVE_ROOT}/rule_reversals.csv', index=False)
    verdicts.to_csv(f'{DRIVE_ROOT}/factor_verdicts.csv', index=False)

    print_cells(means)
    print_orderings(orderings)
    print_verdicts(verdicts)
    print_reversals(reversals)
