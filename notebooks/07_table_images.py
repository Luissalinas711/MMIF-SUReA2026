# Draws three tables from 06_rules_vs_transforms

import matplotlib.pyplot as plt
import pandas as pd

# Readable names for the picture. The CSVs use the short names from the metrics table.
LABELS = {
    'MI': 'mutual information', 'SSIM_mri': 'SSIM to MRI',
    'SSIM_partner': 'SSIM to partner','edge_preservation': 'edge preservation',
    'entropy': 'entropy', 'spatial_freq': 'spatial frequency',
    'std': 'standard deviation',
}

INK = 'black'
RULE = 'gray'
FONT = 11

# Blank space between columns, as a share of the table width
GAP = 0.04

# Never draw narrower than this, however short the text
MIN_WIDTH_INCHES = 9.0

def fit(figure, tables):
    renderer = figure.canvas.get_renderer()
    measured = []
    for headers, rows in tables:
        needed = []
        for column, header in enumerate(headers):
            cells = [header] + [row[column] for row in rows]
            pixels = []
            for index, text in enumerate(cells):
                probe = figure.text(0, 0, text, fontsize=FONT,
                                    weight='bold' if index == 0 else 'normal')
                pixels.append(probe.get_window_extent(renderer).width)
                probe.remove()
            needed.append(max(pixels))
        measured.append(needed)
    widest = max(sum(needed) for needed in measured) / figure.dpi
    figure.set_figwidth(max(MIN_WIDTH_INCHES, widest / (1 - GAP * 6)))

    return [[n / sum(needed) * (1 - GAP * len(needed)) + GAP for n in needed]
            for needed in measured]

def draw(ax, title, headers, rows, widths):
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=FONT + 2, color=INK, pad=12)
    table = ax.table(cellText=rows, colLabels=headers, colWidths=widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(FONT)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD = 0.04
        cell.get_text().set_color(INK)
        cell.get_text().set_ha('left' if column == 0 else 'right')

        if row == 0:
            cell.get_text().set_weight('bold')
            cell.visible_edges = 'B'
            cell.set_edgecolor(RULE)
            cell.set_linewidth(0.8)

# Strongest preference for max-magnitude at the top, so the two columns read as one gradient rather than a list to search through
def ordering_rows(csv_path):
    table = pd.read_csv(csv_path)
    table = table.assign(lean=table['l1'] - table['maxabs']).sort_values('lean')
    headers = ['measure', 'uses sources', r'$\ell_1$', 'max-magnitude', 'significant']
    rows = [[LABELS[row.metric], row.uses_sources, str(row.l1), str(row.maxabs),
             f'{row.significant} of {row.comparisons}'] for row in table.itertuples()]
    return headers, rows

# Sorted by the gap between the smallest rule spread and the largest transform spread,
# which is the quantity the verdict relies on, so the measures where the rule wins sit at the top.
def verdict_rows(csv_path):
    table = pd.read_csv(csv_path)
    table = table.assign(margin=table['rule_min'] - table['transform_max'])
    table = table.sort_values('margin', ascending=False)

    headers = ['measure', 'rule', 'transform', 'dominant', '3 bases only']
    rows = [[LABELS[row.metric],
             f'{row.rule_min:.4f} to {row.rule_max:.4f}',
             f'{row.transform_min:.4f} to {row.transform_max:.4f}',
             row.dominant, row.narrow] for row in table.itertuples()]
    return headers, rows

# The same verdicts inside each pairing, which is the table the review asked for.
# One row per measure, one column per pairing, so the pooled label can be read against them
def pairing_rows(pooled_path, by_pairing_path):
    pooled = pd.read_csv(pooled_path)
    inside = pd.read_csv(by_pairing_path)
    pairings = ['mri_ct', 'mri_spect', 'mri_pet']
    order = (pooled.assign(margin=pooled['rule_min'] - pooled['transform_max'])
                   .sort_values('margin', ascending=False))
    headers = ['measure', 'pooled'] + [p.replace('mri_', 'MRI-').upper() for p in pairings]
    rows = []
    for row in order.itertuples():
        labels = [inside[(inside['metric'] == row.metric)
                         & (inside['pairing'] == p)]['dominant'].iloc[0] for p in pairings]
        rows.append([LABELS[row.metric], row.dominant] + labels)
    return headers, rows

if __name__ == '__main__':
    figure, axes = plt.subplots(3, 1, figsize=(MIN_WIDTH_INCHES, 10))
    figure.subplots_adjust(hspace=0.35)

    titles = [r'Which rule scores higher, $\ell_1$ against max-magnitude',
              'Which factor moves results further, spread when the other is held fixed',
              'The same verdict inside each pairing']

    tables = [ordering_rows(f'{DRIVE_ROOT}/rule_orderings.csv'),
              verdict_rows(f'{DRIVE_ROOT}/factor_verdicts.csv'),
              pairing_rows(f'{DRIVE_ROOT}/factor_verdicts.csv',
                           f'{DRIVE_ROOT}/factor_verdicts_by_pairing.csv')]

    for ax, title, (headers, rows), widths in zip(axes, titles, tables, fit(figure, tables)):
        draw(ax, title, headers, rows, widths)

    out_path = f'{DRIVE_ROOT}/email_tables.png'
    figure.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'written to {out_path}')
