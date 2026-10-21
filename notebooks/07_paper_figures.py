# Draws the two figures the paper prints. Run after 06.
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1a1a1a', '#6b6b6b', '#e3e3e1'
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 7.5, 'axes.linewidth': 0.6})
SHIFTS = [0, 1, 2, 4]

# the paper's names, since the CSVs use the short ones and 'averaging' is the identity row
TRANSFORM_NAMES = {'averaging': 'Identity', 'laplacian': 'Laplacian', 'dwt': 'DWT', 'swt': 'SWT', 'densefuse': 'DenseFuse'}
RULE_NAMES = {'average': 'averaging', 'l1norm': r'$\ell_1$-norm', 'maxabs': 'max-magnitude'}

# Figure 1 shows eight spreads in mean edge preservation. 
# A rule spread varies the rule with  one transform held fixed, 
# a transform spread varies the transform with one rule held fixed
def read_spreads(path):
    cells = pd.read_csv(path)
    edge = cells[cells['metric'] == 'edge_preservation']
    wide = edge.pivot(index='transform', columns='rule', values='value')

    rule = [(TRANSFORM_NAMES[t], wide.loc[t].max() - wide.loc[t].min()) for t in wide.index]
    rule.sort(key=lambda row: -row[1])
    transform = [(RULE_NAMES[r], wide[r].max() - wide[r].min())
                 for r in ('maxabs', 'l1norm', 'average')]
    return rule, transform

# Figure 2 follows the DWT and the SWT under max-magnitude as the partner is shifted
def read_curves(path):
    shifted = pd.read_csv(path)
    rows = shifted[(shifted['metric'] == 'edge_preservation') & (shifted['rule'] == 'maxabs')]
    means = (rows.groupby(['pairing', 'case', 'transform', 'shift_px'],
                          as_index=False)['value'].mean()
             .groupby(['pairing', 'transform', 'shift_px'], as_index=False)['value'].mean())

    curves = {}
    for pairing, label in (('mri_ct', 'MRI-CT'), ('mri_spect', 'MRI-SPECT'),
                           ('mri_pet', 'MRI-PET')):
        block = means[means['pairing'] == pairing]
        curves[label] = {name.upper(): block[block['transform'] == name]
                                       .sort_values('shift_px')['value'].tolist()
                         for name in ('dwt', 'swt')}
    return curves

def bare(ax, sides=('top', 'right')):
    for side in sides:
        ax.spines[side].set_visible(False)

# Written to Drive, not to the working directory
def save(figure, stem):
    folder = Path(DRIVE_ROOT) / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    figure.savefig(folder / f'{stem}.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
    plt.close(figure)
    return folder

def figure_one(RULE, TRANSFORM):
    figure, ax = plt.subplots(figsize=(3.42, 2.20))
    rows = [(name, value, BLUE) for name, value in RULE]
    rows += [(name, value, ORANGE) for name, value in TRANSFORM]
    positions, labels = [], []
    row = 0

    for index, (name, value, colour) in enumerate(reversed(rows)):
        if index == len(TRANSFORM):
            row += 0.8
        ax.barh(row, value, height=0.62, color=colour, zorder=2)
        ax.text(value + 0.009, row, f'{value:.4f}', va='center', fontsize=7, color=INK)
        positions.append(row)
        labels.append(name)
        row += 1
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.40)
    ax.set_ylim(-0.7, row - 0.3)
    ax.set_xlabel('spread in mean edge preservation', fontsize=7.5, color=MUTED)
    bare(ax, ('top', 'right', 'left'))
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], color=BLUE, lw=5, label='rule varied, transform fixed'),
               Line2D([], [], color=ORANGE, lw=5, label='transform varied, rule fixed')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2,
              frameon=False, fontsize=6.8, handlelength=1.1, columnspacing=1.2,
              handletextpad=0.5)
    save(figure, 'fig1')

def figure_two(EDGE):
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 1.80), sharey=True)
    for ax, (pairing, series) in zip(axes, EDGE.items()):
        for name, colour in (('DWT', ORANGE), ('SWT', BLUE)):
            ax.plot(SHIFTS, series[name], color=colour, lw=1.9, marker='o', ms=3.4, zorder=4)
            ax.text(4.1, series[name][-1], f' {name}', fontsize=7, color=colour, va='center')
        ax.set_title(pairing, fontsize=8, color=INK, pad=4)
        ax.set_xticks(SHIFTS)
        ax.set_xlim(-0.2, 5.4)
        ax.set_ylim(0.26, 0.76)
        ax.set_xlabel('shift (pixels)', fontsize=7.5, color=MUTED)
        ax.grid(axis='y', color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        bare(ax)
    axes[0].set_ylabel('mean edge preservation', fontsize=7.5, color=MUTED)
    figure.subplots_adjust(wspace=0.10)

    return save(figure, 'fig2')

if __name__ == '__main__':
    rule, transform = read_spreads(f'{DRIVE_ROOT}/cell_means.csv')
    figure_one(rule, transform)
    folder = figure_two(read_curves(f'{DRIVE_ROOT}/metrics_misalignment.csv'))
    print(f'fig1 and fig2 written as pdf and 600 dpi png to {folder}')
