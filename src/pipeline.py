# Shared fusion process
# Every method notebook was running the exact same three loops (one per modality folder)
# Each notebook just imports its fusion method and runs it through fuse_all_pairs

# This method pairs a transform with a rule, so folders are named like 'laplacian__maxabs'

# For every pair we save two things:
# 1. the fused image on its own
# 2. the MRI , source , fused figure, purely for reference
# Keeping them separate b/c the metrics step later needs the raw fused image

import os

import numpy as np

from utils import load_image_pair, save_fused, display_comparison
from coeffs import make_fusion_function
from transforms import TRANSFORMS
from rules import RULES

# The two images in a pair sit in the same folder and only differ by a piece of the filename, 
# like case21_mri_015.png and case21_ct_015.png, so I find the MRI and swap the '_mri_' part.
# Partner files are named after the TRACER, not the modality. SPECT says 'tc' for technetium tracer and PET
# says 'dg' for FDG tracer, so each folder maps to its tracer
MRI_TOKEN = '_mri_'

MODALITIES = [
    # (subfolder, partner token in the filename, display name)
    ('mri_ct', '_ct_', 'CT'),
    ('mri_spect', '_tc_', 'SPECT'),    
    ('mri_pet', '_dg_', 'PET'),      
]


# Pulls the case number out of a filename so results trace back to a patient
def find_case_number(mri_filename):
    before_token = mri_filename.replace('.png', '').partition(MRI_TOKEN)[0]
    first_piece = before_token.partition('_')[0]

    return first_piece if first_piece.lower().startswith('case') else None


# Shifts the partner sideways and down, for testing how each method holds up when the scans are not lined up.
def shift_partner_image(image, pixels):
    if pixels == 0:
        return image

    height, width = image.shape
    shifted = np.zeros_like(image)
    shifted[pixels:, pixels:] = image[:height - pixels, :width - pixels]

    return shifted


# fusion_function takes mri and partner and returns the fused image
# method_name is the output folder, like 'laplacian__maxabs'
# panel_label titles the fused panel in the comparison figure
# save_figures writes the comparison figures
# show_figures displays them inline in Colab
# skip_finished leaves pairs alone that already have an output
# shift_pixels nudges the partner first, only for the misalignment test
def fuse_all_pairs(fusion_function, method_name, data_root, results_root,
                   panel_label=None, save_figures=False, show_figures=False,
                   skip_finished=True, shift_pixels=0):

    panel_title = panel_label or f'{method_name.title()} Fused'
    pairs_fused = 0
    pairs_skipped = 0

    for subfolder, partner_token, modality in MODALITIES:
        folder = f'{data_root}/{subfolder}'

        # skip the folder if it isn't there
        if not os.path.isdir(folder):
            print(f'  [skip] {folder} not found')
            continue

        # grab every MRI file, the partner gets found by swapping the token below
        mri_files = sorted(f for f in os.listdir(folder)
                           if MRI_TOKEN in f and f.endswith('.png'))

        if not mri_files:
            print(f'  [skip] no MRI files in {subfolder}, run 01_download_images.py first')
            continue

        if find_case_number(mri_files[0]) is None:
            print(f'  [warn] {subfolder}: filenames have no case number, so results cannot be '
                  f'traced back to a patient')

        output_suffix = modality.lower()   # 'CT' -> 'ct', gets placed on the output name
        folder_fused = 0
        folder_skipped = 0

        for mri_file in mri_files:
            partner_file = mri_file.replace(MRI_TOKEN, partner_token)
            output_name = mri_file.replace('.png', '').replace(
                MRI_TOKEN, f'{MRI_TOKEN}{output_suffix}_')

            fused_path = f'{results_root}/{method_name}/fused_only/{output_name}.png'

            if skip_finished and os.path.exists(fused_path):
                folder_skipped += 1
                continue

            # load both as grayscale [0,1], then run the fusion method
            mri, partner = load_image_pair(f'{folder}/{mri_file}',
                                           f'{folder}/{partner_file}')
            partner = shift_partner_image(partner, shift_pixels)
            fused = fusion_function(mri, partner)

            # save the fused image by itself, this is what the metrics read later
            save_fused(fused, fused_path)

            if save_figures:
                display_comparison(
                    mri, partner, fused,
                    titles=['MRI', modality, panel_title],
                    save_path=f'{results_root}/{method_name}/comparisons/{output_name}.png',
                    show=show_figures,
                )

            folder_fused += 1

        pairs_fused += folder_fused
        pairs_skipped += folder_skipped
        print(f'  {subfolder}: {folder_fused} fused, {folder_skipped} already there')

    print(f'[{method_name}] done. {pairs_fused} fused, {pairs_skipped} already there')
    return pairs_fused


# Runs every transform and rule combination through the same fuse_all_pairs
# Goes transform by transform
def run_all_combinations(data_root, results_root, transform_names=None, rule_names=None,
                         shift_pixels=0, skip_finished=True):
    transform_names = transform_names or list(TRANSFORMS)
    rule_names = rule_names or list(RULES)

    print(f'running {len(transform_names)} transforms x {len(rule_names)} rules '
          f'= {len(transform_names) * len(rule_names)} combinations')

    for transform_name in transform_names:
        for rule_name in rule_names:
            method_name = f'{transform_name}__{rule_name}'
            if shift_pixels:
                method_name = f'{method_name}__shift{shift_pixels}'

            fusion_function = make_fusion_function(TRANSFORMS[transform_name], RULES[rule_name])

            fuse_all_pairs(fusion_function, method_name, data_root, results_root,
                           panel_label=f'{transform_name} + {rule_name}',
                           skip_finished=skip_finished, shift_pixels=shift_pixels)


# A free check worth running once the everything finishes
# The average rule is linear, and so are the pyramid and both wavelets.
# Averaging coefficients then transforming back is the same as just averaging the two images, 
# so all four classical transforms have to give the exact same fused image under that rule. 
# If they do not, something is wrong

# DenseFuse is left out b/c the ReLU makes it nonlinear
def check_averaging_column(results_root, tolerance=1e-6):
    import glob

    reference_folder = f'{results_root}/averaging__average/fused_only'
    file_names = sorted(os.path.basename(path)
                        for path in glob.glob(f'{reference_folder}/*.png'))

    if not file_names:
        print('  nothing to check, run the averaging column first')
        return None

    largest_difference = 0.0

    for transform_name in ['laplacian', 'dwt', 'swt']:
        other_folder = f'{results_root}/{transform_name}__average/fused_only'

        for file_name in file_names[:10]:   # a sample is plenty
            reference, other = load_image_pair(f'{reference_folder}/{file_name}',
                                               f'{other_folder}/{file_name}')
            largest_difference = max(largest_difference,
                                     float(np.abs(reference - other).max()))

    verdict = 'ok' if largest_difference < tolerance else 'FAIL, fix before trusting the results'
    print(f'  averaging column check: largest difference {largest_difference:.2e} -> {verdict}')

    return largest_difference# Shared fusion process.
# Every method notebook was running the exact same three loops (one per modality folder).
# Each notebook just imports its fusion method and runs it through fuse_all_pairs.

# This method pairs a transform with a rule, so folders are named like 'laplacian__maxabs'

# For every pair we save two things:
# 1. the fused image on its own
# 2. the MRI , source , fused figure, purely for reference
# Keeping them separate b/c the metrics step later needs the raw fused image

import os

import numpy as np

from utils import load_image_pair, save_fused, display_comparison
from coeffs import make_fusion_function
from transforms import TRANSFORMS
from rules import RULES

# The two images in a pair sit in the same folder and only differ by a piece of the filename, 
# like case21_mri_015.png and case21_ct_015.png, so I find the MRI and swap the '_mri_' part.
# Partner files are named after the TRACER, not the modality. SPECT says 'tc' for technetium tracer and PET
# says 'dg' for FDG tracer, so each folder maps to its tracer
MRI_TOKEN = '_mri_'

MODALITIES = [
    # (subfolder, partner token in the filename, display name)
    ('mri_ct', '_ct_', 'CT'),
    ('mri_spect', '_tc_', 'SPECT'),    
    ('mri_pet', '_dg_', 'PET'),      
]


# Pulls the case number out of a filename so results trace back to a patient
def find_case_number(mri_filename):
    before_token = mri_filename.replace('.png', '').partition(MRI_TOKEN)[0]
    first_piece = before_token.partition('_')[0]

    return first_piece if first_piece.lower().startswith('case') else None


# Shifts the partner sideways and down, for testing how each method holds up when the scans are not lined up.
def shift_partner_image(image, pixels):
    if pixels == 0:
        return image

    height, width = image.shape
    shifted = np.zeros_like(image)
    shifted[pixels:, pixels:] = image[:height - pixels, :width - pixels]

    return shifted


# fusion_function takes mri and partner and returns the fused image
# method_name is the output folder, like 'laplacian__maxabs'
# panel_label titles the fused panel in the comparison figure
# save_figures writes the comparison figures
# show_figures displays them inline in Colab
# skip_finished leaves pairs alone that already have an output
# shift_pixels nudges the partner first, only for the misalignment test
def fuse_all_pairs(fusion_function, method_name, data_root, results_root,
                   panel_label=None, save_figures=False, show_figures=False,
                   skip_finished=True, shift_pixels=0):

    panel_title = panel_label or f'{method_name.title()} Fused'
    pairs_fused = 0
    pairs_skipped = 0

    for subfolder, partner_token, modality in MODALITIES:
        folder = f'{data_root}/{subfolder}'

        # skip the folder if it isn't there
        if not os.path.isdir(folder):
            print(f'  [skip] {folder} not found')
            continue

        # grab every MRI file, the partner gets found by swapping the token below
        mri_files = sorted(f for f in os.listdir(folder)
                           if MRI_TOKEN in f and f.endswith('.png'))

        if not mri_files:
            print(f'  [skip] no MRI files in {subfolder}, run 01_download_images.py first')
            continue

        if find_case_number(mri_files[0]) is None:
            print(f'  [warn] {subfolder}: filenames have no case number, so results cannot be '
                  f'traced back to a patient')

        output_suffix = modality.lower()   # 'CT' -> 'ct', gets placed on the output name
        folder_fused = 0
        folder_skipped = 0

        for mri_file in mri_files:
            partner_file = mri_file.replace(MRI_TOKEN, partner_token)
            output_name = mri_file.replace('.png', '').replace(
                MRI_TOKEN, f'{MRI_TOKEN}{output_suffix}_')

            fused_path = f'{results_root}/{method_name}/fused_only/{output_name}.png'

            if skip_finished and os.path.exists(fused_path):
                folder_skipped += 1
                continue

            # load both as grayscale [0,1], then run the fusion method
            mri, partner = load_image_pair(f'{folder}/{mri_file}',
                                           f'{folder}/{partner_file}')
            partner = shift_partner_image(partner, shift_pixels)
            fused = fusion_function(mri, partner)

            # save the fused image by itself, this is what the metrics read later
            save_fused(fused, fused_path)

            if save_figures:
                display_comparison(
                    mri, partner, fused,
                    titles=['MRI', modality, panel_title],
                    save_path=f'{results_root}/{method_name}/comparisons/{output_name}.png',
                    show=show_figures,
                )

            folder_fused += 1

        pairs_fused += folder_fused
        pairs_skipped += folder_skipped
        print(f'  {subfolder}: {folder_fused} fused, {folder_skipped} already there')

    print(f'[{method_name}] done. {pairs_fused} fused, {pairs_skipped} already there')
    return pairs_fused


# Runs every transform and rule combination through the same fuse_all_pairs
# Goes transform by transform
def run_all_combinations(data_root, results_root, transform_names=None, rule_names=None,
                         shift_pixels=0, skip_finished=True):
    transform_names = transform_names or list(TRANSFORMS)
    rule_names = rule_names or list(RULES)

    print(f'running {len(transform_names)} transforms x {len(rule_names)} rules '
          f'= {len(transform_names) * len(rule_names)} combinations')

    for transform_name in transform_names:
        for rule_name in rule_names:
            method_name = f'{transform_name}__{rule_name}'
            if shift_pixels:
                method_name = f'{method_name}__shift{shift_pixels}'

            fusion_function = make_fusion_function(TRANSFORMS[transform_name], RULES[rule_name])

            fuse_all_pairs(fusion_function, method_name, data_root, results_root,
                           panel_label=f'{transform_name} + {rule_name}',
                           skip_finished=skip_finished, shift_pixels=shift_pixels)


# A free check worth running once the everything finishes
# The average rule is linear, and so are the pyramid and both wavelets.
# Averaging coefficients then transforming back is the same as just averaging the two images, 
# so all four classical transforms have to give the exact same fused image under that rule. 
# If they do not, something is wrong

# DenseFuse is left out b/c the ReLU makes it nonlinear
def check_averaging_column(results_root, tolerance=1e-6):
    import glob

    reference_folder = f'{results_root}/averaging__average/fused_only'
    file_names = sorted(os.path.basename(path)
                        for path in glob.glob(f'{reference_folder}/*.png'))

    if not file_names:
        print('  nothing to check, run the averaging column first')
        return None

    largest_difference = 0.0

    for transform_name in ['laplacian', 'dwt', 'swt']:
        other_folder = f'{results_root}/{transform_name}__average/fused_only'

        for file_name in file_names[:10]:   # a sample is plenty
            reference, other = load_image_pair(f'{reference_folder}/{file_name}',
                                               f'{other_folder}/{file_name}')
            largest_difference = max(largest_difference,
                                     float(np.abs(reference - other).max()))

    verdict = 'ok' if largest_difference < tolerance else 'FAIL, fix before trusting the results'
    print(f'  averaging column check: largest difference {largest_difference:.2e} -> {verdict}')

    return largest_difference
