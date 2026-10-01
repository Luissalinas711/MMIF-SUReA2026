# A quick check over whatever the downloader produced, meant to be run before the fusion grid.

# Two things it checks that verify_image_sizes does not:
#   1. every MRI actually has its partner file sitting next to it
#   2. every filename has a case number in it, so results can be traced back to a patient
# Image sizes are left to verify_image_sizes in utils

import os
from collections import defaultdict

from utils import verify_image_sizes

MRI_TOKEN = '_mri_'

MODALITIES = [
    ('mri_ct', '_ct_', 'CT'),
    ('mri_spect', '_tc_', 'SPECT'),
    ('mri_pet', '_dg_', 'PET'),
]

SLICES_PER_CASE = 3


def verify_dataset(data_root):
    total_pairs = 0
    total_cases = 0
    problems = []

    for subfolder, partner_token, modality in MODALITIES:
        folder = f'{data_root}/{subfolder}'

        if not os.path.isdir(folder):
            print(f'  [skip] {folder} not found')
            continue

        verify_image_sizes(folder)

        mri_files = sorted(f for f in os.listdir(folder)
                           if MRI_TOKEN in f and f.endswith('.png'))
        pairs_per_case = defaultdict(int)
        missing_case_number = 0

        for mri_file in mri_files:
            partner_file = mri_file.replace(MRI_TOKEN, partner_token)

            if not os.path.exists(f'{folder}/{partner_file}'):
                problems.append(f'{subfolder}: {mri_file} has no partner file')
                continue

            case = mri_file.partition(MRI_TOKEN)[0].partition('_')[0]

            # No case number, so it is either left over from the old dataset or the downloader was run with the old naming (just a sanity check)
            if not case.lower().startswith('case'):
                missing_case_number += 1
                continue

            pairs_per_case[case] += 1
            total_pairs += 1

        if missing_case_number:
            problems.append(f'{subfolder}: {missing_case_number} files have no case number, '
                            f'so the old dataset may be mixed in')

        total_cases += len(pairs_per_case)
        print(f'  {subfolder}: {sum(pairs_per_case.values())} pairs from {len(pairs_per_case)} cases')

        short_cases = {case: count for case, count in pairs_per_case.items()
                       if count < SLICES_PER_CASE}
        if short_cases:
            print(f'     cases with fewer than {SLICES_PER_CASE} slices: {short_cases}')

    print(f'\n  total: {total_pairs} pairs across {total_cases} case and pairing slots')
    print(f'  fusion grid will be {total_pairs} x 15 combinations = {total_pairs * 15} fusions')

    if problems:
        print('\n  PROBLEMS')
        for problem in problems:
            print(f'    {problem}')
    else:
        print('\n  no problems found, safe to run the grid')

    return total_pairs


if __name__ == '__main__':
    verify_dataset(DATA)   # DATA comes from the startup snippet
