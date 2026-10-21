# Builds the tables the results section needs:
# the mean for every cell, which rule scores higher on each measure,
# whether either factor dominates, and where the winning rule flips between transforms inside one pairing

# Run after 04. Reads metrics_long.csv for the means and statistics_tests.csv for the orderings.
import re
from itertools import combinations

import pandas as pd
from scipy.stats import spearmanr

TRANSFORMS = ['averaging', 'laplacian', 'dwt', 'swt', 'densefuse']
RULES = ['average', 'l1norm', 'maxabs']
PAIRINGS = ['mri_ct', 'mri_spect', 'mri_pet']

# The identity row applies the rule to raw pixels, which is a pixel-wise max rather than coefficient selection
# thus it sits at the extreme (moslty)
FIXED_BASES = ['laplacian', 'dwt', 'swt']

# Measures that compare the fused image against the sources.
# The other three are computed on the fused image alone.
# This split is for the SWT against DWT result and is not the same split as the rule axis,
# which is why both are printed.
SOURCE_REFERENCED = ['MI', 'SSIM_mri', 'SSIM_partner', 'edge_preservation']

# matches the planned comparisons that hold the transform fixed and vary the rule
WITHIN_TRANSFORM = re.compile(r'(\w+)__l1norm vs \1__maxabs$')
ALPHA = 0.05

# 'averaging' is the transform that works on raw pixels, which the paper calls the identity
# Printing the paper's name in the ordering tables, where 'averaging' would read as the rule
PAPER_NAMES = {'averaging': 'identity', 'laplacian': 'Laplacian', 'dwt': 'DWT',
               'swt': 'SWT', 'densefuse': 'DenseFuse'}

# The four linear transforms give one image under the average rule, so three of the fifteen cells are duplicates
#  Same twelve tests from 04 notebook
DUPLICATE_CELLS = ['laplacian__average', 'dwt__average', 'swt__average']

# A swap only counts as a reordering when the two cells were this far apart, measured as a
# share of the block range. Below it they were near tied and the order never meant much
MEANINGFUL = 0.05

# Li et al. compare multi-resolution transforms, so the three fixed bases are their own class.
# The identity is a control and DenseFuse is learned, and neither is a transform they ranked
TRANSFORM_SETS = {'all five': TRANSFORMS,
                  'without the identity': ['laplacian', 'dwt', 'swt', 'densefuse'],
                  'three fixed bases': FIXED_BASES}

# The partner term in the Table VI model, and the MRI term at the sigmoid ceiling.
# score = ceiling * share + partner * (1 - share), so the crossover is where that line meets the best fused score
SIGMOID_CEILING = 0.9748
PARTNER_TERM = 0.10

# Average each patient's slices, then average over patients, so one patient counts once.
# by_pairing keeps the three apart, b/c every test runs inside one pairing while the pooled table is 22 MRI-SPECT cases out of 35
def cell_means(csv_path, by_pairing=False):
    table = pd.read_csv(csv_path)
    fused = table[table['transform'] != 'unfused']
    per_patient = (fused.groupby(['metric', 'pairing', 'case', 'transform', 'rule'],
                                 as_index=False)['value'].mean())
    keys = ['metric', 'pairing', 'transform', 'rule'] if by_pairing else \
           ['metric', 'transform', 'rule']

    return per_patient.groupby(keys, as_index=False)['value'].mean()

# Vary one factor with the other held fixed. Returns the rule spreads and the transform spreads.
def spreads(block, transforms):
    scoped = block[block['transform'].isin(transforms)]
    return (scoped.groupby('transform')['value'].agg(lambda v: v.max() - v.min()),
            scoped.groupby('rule')['value'].agg(lambda v: v.max() - v.min()))

# One factor dominates only when its smallest spread beats the other's largest 
# This is what makes the claim hold in every cell rather than on average.
# Both tests have to be run. 
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

# The other family 04 writes, averaging against each selection rule inside every transform.
# Same shape as within_transform, one read per family
def post_hoc(tests_path):
    tests = pd.read_csv(tests_path)
    return tests[tests['family'] == 'post hoc']

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

# A ranking fails to carry when the winning rule flips between transforms inside one pairing, 
# so this groups by both rather than pooling them
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

# Two transforms invert when they swap places between the two rules.
# Returns the count, and the count again keeping only pairs that were really apart
def inverted_pairs(wide):
    under_l1, under_maxabs = wide['l1norm'], wide['maxabs']
    span = max(under_l1.max() - under_l1.min(),
               under_maxabs.max() - under_maxabs.min())
    inverted = apart = 0

    for first, second in combinations(wide.index, 2):
        gap_l1 = under_l1[first] - under_l1[second]
        gap_maxabs = under_maxabs[first] - under_maxabs[second]

        if (gap_l1 > 0) == (gap_maxabs > 0):
            continue
        inverted += 1
        if min(abs(gap_l1), abs(gap_maxabs)) / span >= MEANINGFUL:
            apart += 1
    return inverted, apart

# rule_reversals asks whether the better rule changes between transforms. 
# This asks whether the transform order changes between rules, which is the assumption Li et al. work with
# Grouped by pairing for the same reason: pooling the three can agree while each one disagrees
def transform_orderings(means, keep):
    rows = []

    for (metric, pairing), block in means.groupby(['metric', 'pairing']):
        scoped = block[block['transform'].isin(keep)]
        wide = scoped.pivot(index='transform', columns='rule', values='value')

        under_l1 = list(wide['l1norm'].sort_values(ascending=False).index)
        under_maxabs = list(wide['maxabs'].sort_values(ascending=False).index)
        inverted, apart = inverted_pairs(wide)

        rows.append({'metric': metric,
                     'pairing': pairing,
                     'same_order': under_l1 == under_maxabs,
                     'rho': float(spearmanr(wide['l1norm'], wide['maxabs']).statistic),
                     'pairs': len(wide) * (len(wide) - 1) // 2,
                     'inverted': inverted,
                     'inverted_apart': apart,
                     'under_l1': ' > '.join(PAPER_NAMES[t] for t in under_l1),
                     'under_maxabs': ' > '.join(PAPER_NAMES[t] for t in under_maxabs)})

    return pd.DataFrame(rows)

# Every transform set in one frame
def all_transform_orderings(means):
    frames = []
    for label, keep in TRANSFORM_SETS.items():
        table = transform_orderings(means, keep)
        table.insert(0, 'transforms', label)
        frames.append(table)

    return pd.concat(frames, ignore_index=True)


# Counting cells that changed at all is not comparable across the sets
# Not written to a CSV b/c it is a straight sum of transform_orderings.csv
def order_stability(orderings):
    rows = []

    for label, block in orderings.groupby('transforms', sort=False):
        rows.append({'transforms': label,
                     'cells': len(block),
                     'cells_changed': int((~block['same_order']).sum()),
                     'pairs': int(block['pairs'].sum()),
                     'inverted': int(block['inverted'].sum()),
                     'inverted_apart': int(block['inverted_apart'].sum())})

    return pd.DataFrame(rows)

# Both ends of each range, so we can check the verdict from the table ourself.
# by_pairing runs the same test inside each pairing, since the pooled table mixes all three
def factor_verdicts(means, by_pairing=False):
    keys = ['metric', 'pairing'] if by_pairing else ['metric']
    rows = []

    for key, block in means.groupby(keys):
        values = key if isinstance(key, tuple) else (key,)
        rule, transform = spreads(block, TRANSFORMS)

        row = {'metric': values[0]}
        if by_pairing:
            row['pairing'] = values[1]

        row.update({'rule_min': rule.min(),
                    'rule_max': rule.max(),
                    'transform_min': transform.min(),
                    'transform_max': transform.max(),
                    'dominant': verdict(rule, transform),
                    'narrow': verdict(*spreads(block, FIXED_BASES))})
        rows.append(row)
    return pd.DataFrame(rows)

# 04 writes the baseline first in the post hoc family, 
# so a positive median means averaging scored higher and the selection rule fell below it
def post_hoc_summary(post):
    rows = []
    for metric, block in post.groupby('metric'):
        below = block[block['statistic'] > 0]
        share = len(below) / len(block)

        if share >= CLEAR_MAJORITY:
            leans = 'below averaging'
        elif (1 - share) >= CLEAR_MAJORITY:
            leans = 'above averaging'
        else:
            leans = 'split'

        rows.append({'metric': metric,
                     'comparisons': len(block),
                     'significant': (block['p_holm'] < ALPHA).sum(),
                     'below_averaging': len(below),
                     'below_and_significant': (below['p_holm'] < ALPHA).sum(),
                     'leans': leans})
    return pd.DataFrame(rows)

# A measure only counts as leaning if it goes one way 'at least this often'
CLEAR_MAJORITY = 0.75

# The comparisons that run against their own measure's lean. 
# These are the cells the paper has to name when it says committing to one source costs mutual information.
# Takes the summary rather than recomputing the lean, so the threshold is applied in one place
def post_hoc_exceptions(post, summary):
    rows = []

    for measure in summary.itertuples():
        if measure.leans == 'split':
            continue
        block = post[post['metric'] == measure.metric]
        against = (block[block['statistic'] < 0] if measure.leans == 'below averaging'
                   else block[block['statistic'] > 0])
        for row in against.itertuples():
            rows.append({'metric': measure.metric,
                         'pairing': row.pairing,
                         'comparison': row.comparison,
                         'statistic': row.statistic,
                         'p_holm': row.p_holm,
                         'significant': row.p_holm < ALPHA})
    return pd.DataFrame(rows)

# III-F quotes a crossover for each pairing but never the best fused score it is measured against so a reader cannot check it.
# Reads 04's descriptive summary rather than recomputing
def unfused_references(summary_path):
    summary = pd.read_csv(summary_path)
    edge = summary[summary['metric'] == 'edge_preservation']
    rows = []
    for pairing, block in edge.groupby('pairing'):
        fused = block[~block['configuration'].str.startswith('unfused')
                      & ~block['configuration'].isin(DUPLICATE_CELLS)]
        best = fused.loc[fused['mean'].idxmax()]
        mri = float(block[block['configuration'] == 'unfused__mri']['mean'].iloc[0])

        rows.append({'pairing': pairing,
                     'best_fused': float(best['mean']),
                     'best_configuration': best['configuration'],
                     'unfused_mri': mri,
                     'configurations_below_mri': int((fused['mean'] < mri).sum()),
                     'configurations': len(fused),
                     'crossover': (float(best['mean']) - PARTNER_TERM) /
                                  (SIGMOID_CEILING - PARTNER_TERM)})
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

def print_transform_orderings(table):
    print('\n\nwhether the transform order survives a change of rule, within one pairing')
    five = table[table['transforms'] == 'all five']
    for pairing in PAIRINGS:
        block = five[five['pairing'] == pairing]
        print(f'\n  {pairing}')
        for row in block.itertuples():
            if row.same_order:
                print(f'    {row.metric:<20}same order    rho {row.rho:+.3f}')
            else:
                print(f'    {row.metric:<20}DIFFERENT     rho {row.rho:+.3f}')
                print(f'      under l1      {row.under_l1}')
                print(f'      under maxabs  {row.under_maxabs}')
    same = int(five['same_order'].sum())
    print(f'\n  the order survives in {same} of {len(five)} measure and pairing combinations')

def print_order_stability(table):
    print('\n\nhow often the transform order changes, counted by pair')
    for row in table.itertuples():
        rate = row.inverted / row.pairs
        apart_rate = row.inverted_apart / row.pairs
        print(f'  {row.transforms:<22}{row.cells_changed:>3} of {row.cells} cells change, '
              f'{row.inverted:>3} of {row.pairs:>3} pairs invert ({rate:.0%}), '
              f'{row.inverted_apart:>2} by {MEANINGFUL:.0%} or more ({apart_rate:.0%})')

    print('  more transforms means more pairs to invert, so the pair rate is the comparable one')

def print_verdicts_by_pairing(table, pooled):
    print('\n\nthe same test inside each pairing, since the pooled table is mostly MRI-SPECT')
    for pairing in PAIRINGS:
        block = table[table['pairing'] == pairing]
        print(f'\n  {pairing}')
        for row in block.itertuples():
            rule = f'{row.rule_min:.4f} to {row.rule_max:.4f}'
            transform = f'{row.transform_min:.4f} to {row.transform_max:.4f}'
            print(f'    {row.metric:<20}{rule:>20}{transform:>22}{row.dominant:>11}')

    # MRI-PET is left out here b/c three patients cannot support a test either way
    print('\n  does the pooled label hold inside both testable pairings?')
    testable = [p for p in PAIRINGS if p != 'mri_pet']
    print(f'    {"measure":<20}{"pooled":>10}' + ''.join(f'{p:>12}' for p in testable) + '   ')

    for row in pooled.itertuples():
        inside = [table[(table['metric'] == row.metric)
                        & (table['pairing'] == p)]['dominant'].iloc[0] for p in testable]
        note = ('holds' if all(label == row.dominant for label in inside)
                else 'DIFFERS, the paper has to say so')
        print(f'    {row.metric:<20}{row.dominant:>10}'
              + ''.join(f'{label:>12}' for label in inside) + f'   {note}')

def print_post_hoc(table, exceptions):
    print('\n\nthe post hoc family, averaging against each selection rule in every transform')
    print(f'  {"measure":<20}{"significant":>13}{"below averaging":>18}{"both":>8}')
    for row in table.itertuples():
        significant = f'{row.significant} of {row.comparisons}'
        below = f'{row.below_averaging} of {row.comparisons}'
        print(f'  {row.metric:<20}{significant:>13}{below:>18}'
              f'{row.below_and_significant:>8}')

    print('\n  which way each measure leans')
    for row in table.itertuples():
        print(f'    {row.metric:<20}{row.leans}')

    print('\n  the comparisons that run against their own measure')
    if exceptions.empty:
        print('    none')
    for row in exceptions.itertuples():
        mark = ' *' if row.significant else '  '
        print(f'    {row.metric:<20}{row.pairing:<12}{row.comparison:<46}'
              f'median {row.statistic:+.4f}  p_holm {row.p_holm:.4f}{mark}')

def print_unfused(table):
    print('\n\nthe unfused MRI against the best fused score, edge preservation')
    for row in table.itertuples():
        print(f'\n  {row.pairing}')
        print(f'    best fused        {row.best_fused:.4f}  ({row.best_configuration})')
        print(f'    unfused MRI       {row.unfused_mri:.4f}, above '
              f'{row.configurations_below_mri} of {row.configurations} configurations')
        print(f'    crossover implied {row.crossover:.3f}')

if __name__ == '__main__':
    means = cell_means(f'{DRIVE_ROOT}/metrics_long.csv')
    means_by_pairing = cell_means(f'{DRIVE_ROOT}/metrics_long.csv', by_pairing=True)
    within = within_transform(f'{DRIVE_ROOT}/statistics_tests.csv')
    post = post_hoc(f'{DRIVE_ROOT}/statistics_tests.csv')

    orderings = rule_orderings(within)
    reversals = rule_reversals(within)
    verdicts = factor_verdicts(means)

    transforms_by_rule = all_transform_orderings(means_by_pairing)
    stability = order_stability(transforms_by_rule)
    verdicts_by_pairing = factor_verdicts(means_by_pairing, by_pairing=True)
    post_hoc_counts = post_hoc_summary(post)
    exceptions = post_hoc_exceptions(post, post_hoc_counts)
    references = unfused_references(f'{DRIVE_ROOT}/statistics_summary.csv')

    means.to_csv(f'{DRIVE_ROOT}/cell_means.csv', index=False)
    orderings.to_csv(f'{DRIVE_ROOT}/rule_orderings.csv', index=False)
    reversals.to_csv(f'{DRIVE_ROOT}/rule_reversals.csv', index=False)
    verdicts.to_csv(f'{DRIVE_ROOT}/factor_verdicts.csv', index=False)

    means_by_pairing.to_csv(f'{DRIVE_ROOT}/cell_means_by_pairing.csv', index=False)
    transforms_by_rule.to_csv(f'{DRIVE_ROOT}/transform_orderings.csv', index=False)
    verdicts_by_pairing.to_csv(f'{DRIVE_ROOT}/factor_verdicts_by_pairing.csv', index=False)
    post_hoc_counts.to_csv(f'{DRIVE_ROOT}/post_hoc_summary.csv', index=False)
    exceptions.to_csv(f'{DRIVE_ROOT}/post_hoc_exceptions.csv', index=False)
    references.to_csv(f'{DRIVE_ROOT}/unfused_references.csv', index=False)

    print_cells(means)
    print_orderings(orderings)
    print_verdicts(verdicts)
    print_reversals(reversals)

    print_transform_orderings(transforms_by_rule)
    print_order_stability(stability)
    print_verdicts_by_pairing(verdicts_by_pairing, verdicts)
    print_post_hoc(post_hoc_counts, exceptions)
    print_unfused(references)
