# Draws the two figures the paper prints. Run after 06.
# Figure 1 only needs metrics_misalignment.csv. Figure 2 also needs DATA and the fused PNGs 02 left in Drive.

import matplotlib
matplotlib.use('Agg')
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BLUE, ORANGE, INK, GRID = '#2a78d6', '#eb6834', '#1a1a1a', '#d4d4d2'

# The figures are printed 7.24 in wide in the paper, the full text width.
# Heights are set so each new file has the same proportions as the picture it replaces
PRINT_WIDTH = 7.24
CURVES_HEIGHT = PRINT_WIDTH * 1147 / 3509      # 2.37 in
PANELS_HEIGHT = PRINT_WIDTH * 1662 / 3390      # 3.55 in

# Sizes are in points at the printed size. The paper's body text is 10 pt and its captions 8 pt,
# so nothing in a figure should fall below 8
LABEL_SIZE = 9
TICK_SIZE = 8.5
TITLE_SIZE = 9.5

# necessary font for submission (not required for replicating)
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Nimbus Roman', 'Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
                     'font.size': TICK_SIZE, 'axes.linewidth': 0.7, 'pdf.fonttype': 42,
                     'mathtext.fontset': 'stix',
                     'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
SHIFTS = [0, 1, 2, 4]

RULE_NAMES = {'average': 'averaging', 'l1norm': r'$\ell_1$-norm', 'maxabs': 'max-magnitude'}

# 02 wrote the grid here, one folder per cell with the fused images under fused_only
RESULTS = f'{DRIVE_ROOT}/results_grid'

# Figure 2 holds this transform fixed so its two rows can be compared.
# Each case is the one closest to its pairing mean, picked before the images were viewed, and the slice is the clearest of its three. 
# Partner files carry the tracer, NOT the modality
PANEL_TRANSFORM = 'laplacian'
PANEL_ROWS = [('mri_ct', 'case37', 16, '_ct_'),
              ('mri_spect', 'case4', 17, '_tc_')]

# Figure 1 follows the DWT and the SWT under max-magnitude as the partner is shifted
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

# metrics_long holds the slice as a number while the filenames zero pad it, so match on the value rather than rebuilding the padding
def slice_file(folder, case, token, slice_number):
    for path in sorted(Path(folder).glob(f'{case}{token}*.png')):
        tail = path.stem.rpartition('_')[2]
        if tail.isdigit() and int(tail) == slice_number:
            return path
    raise FileNotFoundError(f'no slice {slice_number} for {case}{token} in {folder}')

# Figure 2 puts the two unfused sources beside the three rules, one row per pairing.
# The scores are each slice's own Q, not the case means the tables report
def read_panels(path):
    scored = pd.read_csv(path)
    edge = scored[(scored['metric'] == 'edge_preservation') & (scored['shift_px'] == 0)]
    rows = []
    for pairing, case, slice_number, token in PANEL_ROWS:
        block = edge[(edge['pairing'] == pairing) & (edge['case'] == case)
                     & (edge['slice'] == slice_number)]
        score = block.set_index(['transform', 'rule'])['value']
        suffix = pairing.split('_')[1]   # mri_spect -> spect, which is what 02 put in the name
        sources = f'{DATA}/{pairing}'

        panels = [('MRI', slice_file(sources, case, '_mri_', slice_number), score[('unfused', 'mri')]),
                  (suffix.upper(), slice_file(sources, case, token, slice_number), score[('unfused', 'partner')])]
        for rule in ('average', 'l1norm', 'maxabs'):
            panels.append((RULE_NAMES[rule],
                           slice_file(f'{RESULTS}/{PANEL_TRANSFORM}__{rule}/fused_only', case, f'_mri_{suffix}_', slice_number),
                           score[(PANEL_TRANSFORM, rule)]))
        rows.append((pairing.replace('_', '-').upper(), panels))
    return rows

def bare(ax, sides=('top', 'right')):
    for side in sides:
        ax.spines[side].set_visible(False)

# Written to Drive, not to the working directory.
# No bbox_inches='tight' here: tight cropping changes the shape of the file from run to run
def save(figure, stem):
    folder = Path(DRIVE_ROOT) / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    figure.savefig(folder / f'{stem}.pdf')
    figure.savefig(folder / f'{stem}.png', dpi=600)
    plt.close(figure)
    return folder

# Where the two curves end close together their labels would overlap, so the labels are pushed apart until they are at least min_gap apart
def label_heights(last_values, min_gap):
    names = sorted(last_values, key=last_values.get)
    low, high = names[0], names[1]
    heights = dict(last_values)
    gap = heights[high] - heights[low]
    if gap < min_gap:
        push = (min_gap - gap) / 2
        heights[high] += push
        heights[low] -= push
    return heights

def figure_one(EDGE):
    figure, axes = plt.subplots(1, 3, figsize=(PRINT_WIDTH, CURVES_HEIGHT), sharey=True)
    for ax, (pairing, series) in zip(axes, EDGE.items()):
        heights = label_heights({name: series[name][-1] for name in ('DWT', 'SWT')}, min_gap=0.045)
        for name, colour, marker in (('DWT', ORANGE, 's'), ('SWT', BLUE, 'o')):
            # a different marker for each transform, so the two can be told apart without colour
            ax.plot(SHIFTS, series[name], color=colour, lw=2.0, marker=marker, ms=4.5, zorder=4)
            ax.text(4.22, heights[name], name, fontsize=LABEL_SIZE, color=colour,
                    va='center', fontweight='bold')
        ax.set_title(pairing, fontsize=TITLE_SIZE, pad=5)
        ax.set_xticks(SHIFTS)
        ax.set_xlim(-0.2, 5.3)
        ax.set_ylim(0.26, 0.76)
        ax.set_xlabel('shift (pixels)', fontsize=LABEL_SIZE)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        bare(ax)
    axes[0].set_ylabel('mean edge preservation', fontsize=LABEL_SIZE)
    figure.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.185, wspace=0.08)
    return save(figure, 'fig1')

def figure_two(PANELS):
    figure, axes = plt.subplots(len(PANELS), 5, figsize=(PRINT_WIDTH, PANELS_HEIGHT))
    for row, (label, panels) in zip(axes, PANELS):
        for ax, (name, path, score) in zip(row, panels):
            # matplotlib resamples to the figure dpi for a vector backend, which was shrinking the slices to about 105 px.
            # 'none' keeps the source pixels untouched
            ax.imshow(np.asarray(Image.open(path).convert('L')),
                      cmap='gray', vmin=0, vmax=255, interpolation='none')
            ax.set_title(f'{name}\n{score:.4f}', fontsize=LABEL_SIZE, pad=3, linespacing=1.15)
            ax.set_xticks([])
            ax.set_yticks([])
            bare(ax, ('top', 'right', 'bottom', 'left'))
        row[0].set_ylabel(label, fontsize=TITLE_SIZE, labelpad=6)
    figure.subplots_adjust(left=0.035, right=0.998, top=0.915, bottom=0.005,
                           wspace=0.04, hspace=0.24)

    return save(figure, 'fig2')

if __name__ == '__main__':
    figure_one(read_curves(f'{DRIVE_ROOT}/metrics_misalignment.csv'))
    folder = figure_two(read_panels(f'{DRIVE_ROOT}/metrics_long.csv'))
    print(f'fig1 and fig2 written as pdf and 600 dpi png to {folder}')
