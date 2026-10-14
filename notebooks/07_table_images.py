# Draws two tables from 06_rules_vs_transforms

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


# Largest ratio first, so the measures where the rule wins sit at the top.
def verdict_rows(csv_path):
    table = pd.read_csv(csv_path).sort_values('ratio', ascending=False)
    headers = ['measure', 'rule', 'transform', 'ratio', 'dominant', '3 bases only']
    rows = [[LABELS[row.metric], f'{row.rule:.4f}', f'{row.transform:.4f}', f'{row.ratio:.2f}',
             row.dominant, row.narrow] for row in table.itertuples()]
    return headers, rows


if __name__ == '__main__':
    figure, axes = plt.subplots(2, 1, figsize=(9, 7))
    figure.subplots_adjust(hspace=0.3)

    headers, rows = ordering_rows(f'{DRIVE_ROOT}/rule_orderings.csv')
    draw(axes[0], r'Which rule scores higher, $\ell_1$ against max-magnitude',
         headers, rows, [0.30, 0.16, 0.14, 0.22, 0.18])

    headers, rows = verdict_rows(f'{DRIVE_ROOT}/factor_verdicts.csv')
    draw(axes[1], 'Which factor moves results further',
         headers, rows, [0.28, 0.13, 0.15, 0.11, 0.16, 0.17])

    out_path = f'{DRIVE_ROOT}/email_tables.png'
    figure.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'written to {out_path}')
