# Score every method's fused images and render clean table images
# Run the startup cell first

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from metrics import all_metrics

RESULTS = f'{REPO_PATH}/results'
TABLES = f'{RESULTS}/tables'
os.makedirs(TABLES, exist_ok=True)

# method order
METHODS = ['averaging', 'densefuse', 'densefuse_norm', 'dwt', 'laplacian']

# metrics to show, in order, with names for the table headers
METRICS = ['entropy', 'MI', 'std', 'spatial_freq',
           'SSIM_mri', 'SSIM_partner', 'edge_preservation']
METRIC_LABELS = {
    'entropy':           'Entropy',
    'MI':                'MI',
    'std':               'Std dev',
    'spatial_freq':      'Spatial freq',
    'SSIM_mri':          'SSIM (MRI)',
    'SSIM_partner':      'SSIM (partner)',
    'edge_preservation': 'Edge pres.',
}
METHOD_LABELS = {
    'averaging':      'Averaging',
    'densefuse':      'DenseFuse (add)',
    'densefuse_norm': 'DenseFuse (norm)',
    'dwt':            'DWT',
    'laplacian':      'Laplacian',
}


def load_gray_image(path):
    # Load as grayscale in [0,1] - the same way every time, so numbers compare.
    return np.array(Image.open(path).convert('L')).astype('float32') / 255.0


def find_partner(mri_path):
    # The partner sits in the same folder; swap the modality chunk to find it.
    folder = os.path.dirname(mri_path)
    mri_name = os.path.basename(mri_path)
    for chunk in ['_ct_', '_tc_', '_dg_', '_pet_', '_spect_']:
        partner_path = os.path.join(folder, mri_name.replace('_mri_', chunk))
        if os.path.exists(partner_path):
            return partner_path
    return None


def fused_name_for(mri_name, pairing):
    # Standardized fused name
    modality = pairing.split('_', 1)[1]        # 'mri_ct' -> 'ct'
    return mri_name.replace('_mri_', f'_mri_{modality}_')


def evaluate():
    rows = []
    # sources live in mri_ct/ , mri_pet/ , mri_spect/ subfolders
    mri_paths = sorted(glob.glob(f'{DATA}/**/*_mri_*.png', recursive=True))

    for mri_path in mri_paths:
        partner_path = find_partner(mri_path)
        if partner_path is None:
            print(f'  no partner for {os.path.basename(mri_path)}, skipping')
            continue

        mri = load_gray_image(mri_path)
        partner = load_gray_image(partner_path)
        mri_name = os.path.basename(mri_path)
        pairing = os.path.basename(os.path.dirname(mri_path))
        fused_name = fused_name_for(mri_name, pairing)

        for method in METHODS:
            fused_path = f'{RESULTS}/{method}/fused_only/{fused_name}'
            if not os.path.exists(fused_path):
                print(f'  missing fused: {fused_name} for {method}')
                continue

            fused = load_gray_image(fused_path)
            row = all_metrics(mri, partner, fused)
            row['method'] = method
            row['pairing'] = pairing
            row['pair'] = fused_name
            rows.append(row)

    return pd.DataFrame(rows)


def save_table_image(scores, title, filename):
    # Render a metrics table (methods x metrics) as a clean image.
    # The highest value in each metric column is highlighted
    header = ['Method'] + [METRIC_LABELS[metric] for metric in scores.columns]
    cell_text = [
        [METHOD_LABELS[method]] + [f'{scores.loc[method, metric]:.4f}'
                                   for metric in scores.columns]
        for method in scores.index
    ]

    n_rows = len(scores.index)
    n_cols = len(header)
    fig, ax = plt.subplots(figsize=(2 + n_cols * 1.4, 1.2 + n_rows * 0.5))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    table = ax.table(cellText=cell_text, colLabels=header, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    table.auto_set_column_width(col=list(range(n_cols)))

    # row index (in the table) of the highest value for each metric column
    best_row = {metric: int(scores[metric].values.argmax()) + 1 for metric in scores.columns}

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row == 0:                       # header
            cell.set_facecolor('steelblue')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 0:                     # method names
            cell.set_facecolor('aliceblue')
            cell.set_text_props(fontweight='bold')
        else:                              # data cells
            cell.set_facecolor('white' if row % 2 else 'whitesmoke')
            metric = scores.columns[col - 1]
            if best_row[metric] == row:    # highlight the highest value
                cell.set_facecolor('honeydew')
                cell.set_text_props(fontweight='bold')

    fig.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.show()
    print(f'saved -> {filename}')


def pairing_title(pairing):
    # 'mri_ct' -> 'MRI-SPECT'-style label
    return 'MRI\u2013' + pairing.split('_', 1)[1].upper()


table = evaluate()
if table.empty:
    raise SystemExit('No fused images matched. Check DATA, RESULTS, and the naming.')

print(f'Evaluated {len(table)} fused images across {table["method"].nunique()} methods.\n')

# average each metric per method, keep the order
summary = table.groupby('method')[METRICS].mean().reindex(METHODS)

# save the numbers as CSVs (full detail + summary)
table.to_csv(f'{RESULTS}/metrics_per_pair.csv', index=False)
summary.round(4).to_csv(f'{RESULTS}/metrics_summary.csv')

# rendered table images
save_table_image(summary, 'Average metrics per method',
                 f'{TABLES}/summary_table.png')

for pairing in sorted(table['pairing'].unique()):
    pairing_scores = table[table['pairing'] == pairing].groupby('method')[METRICS].mean()
    pairing_scores = pairing_scores.reindex(METHODS)
    save_table_image(pairing_scores, f'{pairing_title(pairing)} pairs',
                     f'{TABLES}/table_{pairing}.png')

print(f'\nTable images saved under {TABLES}/')
