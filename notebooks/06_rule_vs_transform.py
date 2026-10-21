# Builds the tables the results section needs:
# the mean for every cell, which rule scores higher on each measure, whether either factor dominates,
# and where the winning rule flips between transforms inside one pairing

# Run after 04. Reads metrics_long.csv for the means and statistics_tests.csv for the orderings.
import re
import pandas as pd
from scipy.stats import spearmanr

TRANSFORMS = ['averaging', 'laplacian', 'dwt', 'swt', 'densefuse']
RULES = ['average', 'l1norm', 'maxabs']
PAIRINGS = ['mri_ct', 'mri_spect', 'mri_pet']

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

# 'averaging' is the transform that works on raw pixels, which the paper calls the identity.
# Printing the paper's name in the ordering tables, where 'averaging' would read as the rule
PAPER_NAMES = {'averaging': 'identity', 'laplacian': 'Laplacian', 'dwt': 'DWT',
               'swt': 'SWT', 'densefuse': 'DenseFuse'}

# Table III and the Table IV edge preservation row, as printed in the draft.
# Checked on every run
PUBLISHED_EDGE_MEANS = {
    'averaging': (0.4212, 0.6624, 0.7062), 'laplacian': (0.4212, 0.7255, 0.7428),
    'dwt': (0.4212, 0.5998, 0.6216), 'swt': (0.4212, 0.6577, 0.6897),
    'densefuse': (0.4165, 0.6890, 0.7106),
}
PUBLISHED_EDGE_SPREADS = (0.2004, 0.3216, 0.0047, 0.1257)

# The four linear transforms give one image under the average rule, so three of the fifteen cells are duplicates
DUPLICATE_CELLS = ['laplacian__average', 'dwt__average', 'swt__average']

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
# Both tests have to be run
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

# A ranking fails to carry when the winning rule flips between transforms inside one pairing
# this groups by both rather than pooling them
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

# The other direction of the same question. rule_reversals asks whether the better rule changes
# between transforms. This asks whether the transform order changes between rules, which is the
# assumption Li et al. carry forward.
# Grouped by pairing for the same reason: pooling the three can agree while each one disagrees
def transform_orderings(means):
    rows = []

    for (metric, pairing), block in means.groupby(['metric', 'pairing']):
        wide = block.pivot(index='transform', columns='rule', values='value')
        under_l1 = list(wide['l1norm'].sort_values(ascending=False).index)
        under_maxabs = list(wide['maxabs'].sort_values(ascending=False).index)

        rows.append({'metric': metric,
                     'pairing': pairing,
                     'same_order': under_l1 == under_maxabs,
                     'rho': float(spearmanr(wide['l1norm'], wide['maxabs']).statistic),
                     'under_l1': ' > '.join(PAPER_NAMES[t] for t in under_l1),
                     'under_maxabs': ' > '.join(PAPER_NAMES[t] for t in under_maxabs)})

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

# How often the pooled label survives inside the two pairings that carry tests.
# MRI-PET is left out b/c three patients cannot support a test either way
def verdicts_agree(pooled, by_pairing):
    testable = [p for p in PAIRINGS if p != 'mri_pet']
    rows = []
    for row in pooled.itertuples():
        inside = {p: by_pairing[(by_pairing['metric'] == row.metric)
                                & (by_pairing['pairing'] == p)]['dominant'].iloc[0]
                  for p in testable}
        rows.append({'metric': row.metric,
                     'pooled': row.dominant,
                     **inside,
                     'holds': all(label == row.dominant for label in inside.values())})
    return pd.DataFrame(rows)

# 04 writes the baseline first in the post hoc family, so a positive median means averaging
# scored higher and the selection rule fell below it
def post_hoc_summary(post):
    rows = []
    for metric, block in post.groupby('metric'):
        below = block[block['statistic'] > 0]
        rows.append({'metric': metric,
                     'comparisons': len(block),
                     'significant': (block['p_holm'] < ALPHA).sum(),
                     'below_averaging': len(below),
                     'below_and_significant': (below['p_holm'] < ALPHA).sum()})
    return pd.DataFrame(rows)

# A measure only has exceptions if it has a pattern to break. Most measures sit well above
# averaging, which is the expected direction, so listing those is noise.
# This keeps the measures that lean one way at least this often, then lists the few that do not
CLEAR_MAJORITY = 0.75

# The handful of comparisons running against their own measure's direction. These are the cells
# the paper has to exclude when it says committing to one source costs mutual information
def post_hoc_exceptions(post):
    rows = []
    for metric, block in post.groupby('metric'):
        below = block['statistic'] > 0
        share = below.sum() / len(block)

        if share >= CLEAR_MAJORITY:
            against, pattern = block[~below], 'below averaging'
        elif (1 - share) >= CLEAR_MAJORITY:
            against, pattern = block[below], 'above averaging'
        else:
            continue                      # split down the middle, so nothing to be an exception to

        for row in against.itertuples():
            rows.append({'metric': metric,
                         'pattern': pattern,
                         'pairing': row.pairing,
                         'comparison': row.comparison,
                         'statistic': row.statistic,
                         'p_holm': row.p_holm,
                         'significant': row.p_holm < ALPHA})
    return pd.DataFrame(rows)

# Which measures lean clearly, and which are too split to describe as leaning at all
def post_hoc_patterns(post):
    rows = []
    for metric, block in post.groupby('metric'):
        share = (block['statistic'] > 0).sum() / len(block)
        if share >= CLEAR_MAJORITY:
            pattern = 'below averaging'
        elif (1 - share) >= CLEAR_MAJORITY:
            pattern = 'above averaging'
        else:
            pattern = 'split, no clear direction'
        rows.append({'metric': metric, 'pattern': pattern})
    return pd.DataFrame(rows)

# III-F quotes a crossover for each pairing but never the best fused score it is measured against so it cannot be checked
# Reads my notebook 04's descriptive summary instead of recomputing
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

# like check_averaging_column.
# Tables III and IV came out of this file, so if the means or the spread code move, the paper will stop matching the code
def check_published_numbers(means):
    edge = means[means['metric'] == 'edge_preservation']
    wide = edge.pivot(index='transform', columns='rule', values='value')
    # the paper prints four decimals,
    mismatches = []

    for transform, published in PUBLISHED_EDGE_MEANS.items():
        for rule, value in zip(RULES, published):
            got = round(float(wide.loc[transform, rule]), 4)
            if got != value:
                mismatches.append(f'{transform} {rule}: {got} not {value}')
    rule_spread, transform_spread = spreads(edge, TRANSFORMS)
    computed = (rule_spread.min(), rule_spread.max(),
                transform_spread.min(), transform_spread.max())
    labels = ['rule min', 'rule max', 'transform min', 'transform max']

    for label, got, published in zip(labels, computed, PUBLISHED_EDGE_SPREADS):
        if round(float(got), 4) != published:
            mismatches.append(f'{label}: {round(float(got), 4)} not {published}')

    if mismatches:
        print('\npublished Table III and IV check: FAIL, the paper no longer matches the code')
        for line in mismatches:
            print(f'    {line}')
    else:
        print('\npublished Table III and IV check: all 19 printed values reproduce, ok')
    return mismatches

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
    for pairing in PAIRINGS:
        block = table[table['pairing'] == pairing]
        print(f'\n  {pairing}')
        for row in block.itertuples():
            if row.same_order:
                print(f'    {row.metric:<20}same order    rho {row.rho:+.3f}')
            else:
                print(f'    {row.metric:<20}DIFFERENT     rho {row.rho:+.3f}')
                print(f'      under l1      {row.under_l1}')
                print(f'      under maxabs  {row.under_maxabs}')

    same = int(table['same_order'].sum())
    print(f'\n  the order survives in {same} of {len(table)} measure and pairing combinations')

def print_verdicts_by_pairing(table, agreement):
    print('\n\nthe same test inside each pairing, since the pooled table is mostly MRI-SPECT')
    for pairing in PAIRINGS:
        block = table[table['pairing'] == pairing]
        print(f'\n  {pairing}')
        for row in block.itertuples():
            rule = f'{row.rule_min:.4f} to {row.rule_max:.4f}'
            transform = f'{row.transform_min:.4f} to {row.transform_max:.4f}'
            print(f'    {row.metric:<20}{rule:>20}{transform:>22}{row.dominant:>11}')
    print('\n  does the pooled label hold inside both testable pairings?')
    testable = [p for p in PAIRINGS if p != 'mri_pet']
    print(f'    {"measure":<20}{"pooled":>10}' + ''.join(f'{p:>12}' for p in testable) + '   ')
    for row in agreement.itertuples():
        labels = ''.join(f'{getattr(row, p):>12}' for p in testable)
        note = 'holds' if row.holds else 'DIFFERS, the paper has to say so'
        print(f'    {row.metric:<20}{row.pooled:>10}{labels}   {note}')

def print_post_hoc(table, patterns, exceptions):
    print('\n\nthe post hoc family, averaging against each selection rule in every transform')
    print(f'  {"measure":<20}{"significant":>13}{"below averaging":>18}{"both":>8}')
    for row in table.itertuples():
        significant = f'{row.significant} of {row.comparisons}'
        below = f'{row.below_averaging} of {row.comparisons}'
        print(f'  {row.metric:<20}{significant:>13}{below:>18}'
              f'{row.below_and_significant:>8}')
    print('\n  which way each measure leans')
    for row in patterns.itertuples():
        print(f'    {row.metric:<20}{row.pattern}')
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

    transforms_by_rule = transform_orderings(means_by_pairing)
    verdicts_by_pairing = factor_verdicts(means_by_pairing, by_pairing=True)
    agreement = verdicts_agree(verdicts, verdicts_by_pairing)
    post_hoc_counts = post_hoc_summary(post)
    exceptions = post_hoc_exceptions(post)
    patterns = post_hoc_patterns(post)
    references = unfused_references(f'{DRIVE_ROOT}/statistics_summary.csv')

    means.to_csv(f'{DRIVE_ROOT}/cell_means.csv', index=False)
    orderings.to_csv(f'{DRIVE_ROOT}/rule_orderings.csv', index=False)
    reversals.to_csv(f'{DRIVE_ROOT}/rule_reversals.csv', index=False)
    verdicts.to_csv(f'{DRIVE_ROOT}/factor_verdicts.csv', index=False)

    means_by_pairing.to_csv(f'{DRIVE_ROOT}/cell_means_by_pairing.csv', index=False)
    transforms_by_rule.to_csv(f'{DRIVE_ROOT}/transform_orderings.csv', index=False)
    verdicts_by_pairing.to_csv(f'{DRIVE_ROOT}/factor_verdicts_by_pairing.csv', index=False)
    post_hoc_counts.to_csv(f'{DRIVE_ROOT}/post_hoc_summary.csv', index=False)
    references.to_csv(f'{DRIVE_ROOT}/unfused_references.csv', index=False)

    print_cells(means)
    print_orderings(orderings)
    print_verdicts(verdicts)
    print_reversals(reversals)

    print_transform_orderings(transforms_by_rule)
    print_verdicts_by_pairing(verdicts_by_pairing, agreement)
    print_post_hoc(post_hoc_counts, patterns, exceptions)
    print_unfused(references)

    check_published_numbers(means)
