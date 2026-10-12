# Scores fused images against their two sources and writes one long CSV
# pipeline.py runs the fusing, this script runs the measuring
# One row per measurement, with the case on every row so the statistics can group by patient

import csv
import os
import numpy as np
from PIL import Image as PILImage
from metrics import all_metrics
from pipeline import MODALITIES, MRI_TOKEN
from utils import load_image_pair

# the CSV columns, and the keys metric_rows has to produce
FIELDS = ['case', 'pairing', 'slice', 'transform', 'rule', 'shift_px', 'metric', 'value']

def load_grayscale(path):
    return np.array(PILImage.open(path).convert('L'), dtype=np.float32) / 255.0

# laplacian__maxabs -> ('laplacian', 'maxabs', 0), swt__l1norm__shift2 -> ('swt', 'l1norm', 2)
def split_method(method_name):
    parts = method_name.split('__')
    shift = int(parts[2].replace('shift', '')) if len(parts) > 2 else 0
    return parts[0], parts[1], shift

# Shifting leaves a black band on the top and left. 
# Crop every image by the same margin, shift 0 included, so the numbers compare fusion quality and not border size.
def crop(image, margin):
    return image[margin:, margin:]

# case03_mri_007.png -> ('case03', '007', 'case03_mri_ct_007.png'), the name 02 saved it under
def pair_names(mri_file, suffix):
    stem = mri_file.replace('.png', '')
    case = stem.partition(MRI_TOKEN)[0]
    slice_number = stem.rpartition('_')[2]
    fused_name = stem.replace(MRI_TOKEN, f'{MRI_TOKEN}{suffix}_') + '.png'

    return case, slice_number, fused_name


# Seven rows, one per metric. The fused images and the unfused references both come through here
def metric_rows(case, pairing, slice_number, transform, rule, shift, mri, partner, fused):
    return [{'case': case,
             'pairing': pairing,
             'slice': slice_number,
             'transform': transform,
             'rule': rule,
             'shift_px': shift,
             'metric': metric_name,
             'value': value}
            for metric_name, value in all_metrics(mri, partner, fused).items()]


def score_everything(data_root, results_root, output_path, crop_margin=0):
    combinations = sorted(name for name in os.listdir(results_root) if '__' in name)
    print(f'{len(combinations)} combinations found')

    rows = []
    missing = []

    for subfolder, partner_token, modality in MODALITIES:
        folder = f'{data_root}/{subfolder}'

        if not os.path.isdir(folder):
            print(f'  [skip] {folder} not found')
            continue

        mri_files = sorted(f for f in os.listdir(folder)
                           if MRI_TOKEN in f and f.endswith('.png'))

        for mri_file in mri_files:
            partner_file = mri_file.replace(MRI_TOKEN, partner_token)

            # sources load once per pair and stay unshifted, b/c the question is how much quality is lost against a correctly aligned pair
            mri, partner = load_image_pair(f'{folder}/{mri_file}',
                                           f'{folder}/{partner_file}')
            mri = crop(mri, crop_margin)
            partner = crop(partner, crop_margin)
            case, slice_number, fused_name = pair_names(mri_file, modality.lower())

            for combination in combinations:
                fused_path = f'{results_root}/{combination}/fused_only/{fused_name}'
                if not os.path.exists(fused_path):
                    missing.append(f'{combination}/{fused_name}')
                    continue
                transform, rule, shift = split_method(combination)
                fused = crop(load_grayscale(fused_path), crop_margin)
                rows.extend(metric_rows(case, subfolder, slice_number, transform, rule, shift,
                                        mri, partner, fused))

            # the unfused sources, scored the same way as everything else
            for rule, source in (('mri', mri), ('partner', partner)):
                rows.extend(metric_rows(case, subfolder, slice_number, 'unfused', rule, 0,
                                        mri, partner, source))

        print(f'  {subfolder}: {len(mri_files)} pairs scored')

    with open(output_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\n{len(rows)} rows written to {output_path}')
    if missing:
        print(f'  {len(missing)} fused images missing, first few: {missing[:3]}')
    return rows
