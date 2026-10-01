# Downloads MRI+CT, MRI+SPECT and MRI+PET pairs from the Harvard Whole Brain Atlas
# Saved as PNG regardless of what the server holds, since some cases are GIF
# Case lists came from a quick scan of the whole atlas for cases with both modalities at the same slice
# PET only has three cases in the atlas

import os
import time
import urllib.request

import numpy as np
from PIL import Image as PILImage

BASE_URL = 'http://www.med.harvard.edu/AANLIB/cases'

# URLs look like BASE_URL/case21/ct1/015.gif
FOLDER_NUMBERS = ['1', '2', '3', '4']
FILE_EXTENSIONS = ['.gif', '.png']

MRI_CODE = 'mr'
ALL_SLICE_NUMBERS = [f'{i:03d}' for i in range(1, 76)]

SECONDS_BETWEEN_REQUESTS = 0.05      # old server, space the requests out

# Three per case, not five. Neighbouring slices are nearly the same image
SLICES_PER_CASE = 3

# Skip the outer 15% of each case, where slices are mostly empty 
EDGE_MARGIN = 0.15

MAX_ASPECT_RATIO = 1.5               

# Anatomical and functional need different thresholds. 
# MRI and CT fill the frame with tissue, 
# SPECT and PET are mostly black by design with signal in a few regions
# Direction matters: dropping low signal SPECT would leave only the brightest, which is where fusion wash out is weakest, 
# so it would hide the effect being measured

ANATOMICAL_CODES = ('mr', 'ct')
FUNCTIONAL_CODES = ('tc', 'dg')

MIN_BRAIN_FRACTION = 0.15
MIN_BRAIN_CONTRAST = 15.0

MIN_SIGNAL_FRACTION = 0.02
MIN_SIGNAL_CONTRAST = 5.0

# Partner files are named by TRACER, not modality: tc is technetium, dg is FDG.
DOWNLOAD_PLAN = [
    ('mri_ct', 'ct', [
        'case2', 'case16', 'case20', 'case21', 'case28',
        'case32', 'case33', 'case34', 'case37', 'case41',
    ]),
    ('mri_spect', 'tc', [
        'case1', 'case3', 'case4', 'case9', 'case10', 'case11', 'case12',
        'case14', 'case15', 'case17', 'case18', 'case19', 'case21', 'case22',
        'case24', 'case25', 'case28', 'case29', 'case31', 'case35', 'case36',
        'case40',
    ]),
    ('mri_pet', 'dg', [
        'caseNN1', 'caseSLU', 'caseSLU2',
    ]),
]

# For figure captions only. Filenames use the case number so images trace back to a patient.
CASE_LABELS = {
    'case3': 'alzheimers',
    'case11': 'huntingtons',
    'case18': 'vascular_dementia',
    'case25': 'encephalitis',
    'case28': 'carcinoma',
    'case41': 'toxoplasmosis',
    'caseNN1': 'alzheimers_mild',
    'caseSLU': 'glioma',
    'caseSLU2': 'glioma_2',
}

rejected_images = []      # (case, slice, modality code, reason)


# Returns a reason to throw the image out or None
def find_quality_problem(image_path, modality_code):
    try:
        image = PILImage.open(image_path).convert('L')
    except Exception:
        return 'corrupt_or_truncated'

    width, height = image.size
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        return 'montage_or_strip'

    pixels = np.array(image)

    # not pure black background, not pure white edge. in between so good to go
    content_fraction = ((pixels > 25) & (pixels < 245)).mean()
    contrast = pixels.std()

    if modality_code in ANATOMICAL_CODES:
        if content_fraction < MIN_BRAIN_FRACTION:
            return 'not_enough_brain'
        if contrast < MIN_BRAIN_CONTRAST:
            return 'flat_or_washed_out'
    else:
        if content_fraction < MIN_SIGNAL_FRACTION:
            return 'no_signal'
        if contrast < MIN_SIGNAL_CONTRAST:
            return 'flat_or_washed_out'

    return None


# Tries every folder number and extension until one downloads and passes.
def download_one_image(case, modality_code, slice_number, save_path):
    for folder_number in FOLDER_NUMBERS:
        for extension in FILE_EXTENSIONS:
            url = f'{BASE_URL}/{case}/{modality_code}{folder_number}/{slice_number}{extension}'
            temp_path = save_path.replace('.png', f'_temp{extension}')

            try:
                urllib.request.urlretrieve(url, temp_path)
            except Exception:
                time.sleep(SECONDS_BETWEEN_REQUESTS)
                continue

            problem = find_quality_problem(temp_path, modality_code)

            if problem is None:
                image = PILImage.open(temp_path).convert('L')
                if image.size != (256, 256):
                    image = image.resize((256, 256), PILImage.LANCZOS)
                image.save(save_path)
                os.remove(temp_path)
                return True

            rejected_images.append((case, slice_number, modality_code, problem))
            os.remove(temp_path)
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    return False


# Which slices exist, using HEAD so nothing downloads yet.
def find_available_slices(case, modality_code):
    for folder_number in FOLDER_NUMBERS:
        for extension in FILE_EXTENSIONS:
            found_slices = []

            for slice_number in ALL_SLICE_NUMBERS:
                url = (f'{BASE_URL}/{case}/{modality_code}{folder_number}'
                       f'/{slice_number}{extension}')
                try:
                    request = urllib.request.Request(url, method='HEAD')
                    with urllib.request.urlopen(request, timeout=10) as response:
                        if response.status == 200:
                            found_slices.append(slice_number)
                except Exception:
                    pass
                time.sleep(SECONDS_BETWEEN_REQUESTS)

            if found_slices:
                return found_slices

    return []


# Spread across the middle instead of taking the first few in a row
def pick_spread_out_slices(available_slices, how_many=SLICES_PER_CASE, margin=EDGE_MARGIN):
    if len(available_slices) <= how_many:
        return list(available_slices)

    first = int(len(available_slices) * margin)
    last = int(len(available_slices) * (1 - margin))
    middle_slices = available_slices[first:last] or available_slices

    step = len(middle_slices) / (how_many + 1)
    return [middle_slices[int(step * (position + 1))] for position in range(how_many)]


def download_all_pairs(data_root):
    for folder_name, modality_code, case_list in DOWNLOAD_PLAN:
        folder = f'{data_root}/{folder_name}'
        os.makedirs(folder, exist_ok=True)
        print(f'\n{folder_name}: {len(case_list)} cases')

        pairs_saved = 0

        for case in case_list:
            mri_slices = set(find_available_slices(case, MRI_CODE))
            partner_slices = set(find_available_slices(case, modality_code))

            # a pair needs both scans at the same slice number
            matching_slices = sorted(mri_slices & partner_slices)

            if len(matching_slices) < SLICES_PER_CASE:
                print(f'  {case}: only {len(matching_slices)} matching slices, skipping')
                continue

            chosen_slices = pick_spread_out_slices(matching_slices)

            for slice_number in chosen_slices:
                mri_path = f'{folder}/{case}_mri_{slice_number}.png'
                partner_path = f'{folder}/{case}_{modality_code}_{slice_number}.png'

                # already have it, so a re-run gets anything missing
                if os.path.exists(mri_path) and os.path.exists(partner_path):
                    pairs_saved += 1
                    continue

                got_mri = download_one_image(case, MRI_CODE, slice_number, mri_path)
                got_partner = download_one_image(case, modality_code, slice_number, partner_path)

                if got_mri and got_partner:
                    pairs_saved += 1
                else:
                    # never leave half a pair, a lone MRI breaks the fusion step
                    for path in (mri_path, partner_path):
                        if os.path.exists(path):
                            os.remove(path)

            print(f'  {case}: {chosen_slices} -> running total {pairs_saved}')

        print(f'  {folder_name} done: {pairs_saved} pairs')

    save_summary(data_root)


# What ended up on disk and what got thrown out, so I can check later without re-downloading.
def save_summary(data_root):
    summary_path = f'{data_root}/download_summary.txt'

    with open(summary_path, 'w') as summary_file:
        summary_file.write('MMIF fusion dataset\n')
        summary_file.write(f'{SLICES_PER_CASE} slices per case, spread across the middle\n\n')

        for folder_name, modality_code, case_list in DOWNLOAD_PLAN:
            files = sorted(f for f in os.listdir(f'{data_root}/{folder_name}')
                           if f.endswith('.png'))
            cases_present = {f.split('_')[0] for f in files}

            summary_file.write(f'{folder_name}: {len(files) // 2} pairs '
                               f'from {len(cases_present)} cases\n')
            for file_name in files:
                summary_file.write(f'  {file_name}\n')

        summary_file.write(f'\nrejected during this run only: {len(rejected_images)}\n')

        reasons = {}
        for _, _, _, reason in rejected_images:
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
            summary_file.write(f'  {reason}: {count}\n')

    print(f'\nsummary written to {summary_path}')
    print('Do not commit the images, the atlas has its own licensing.')


if __name__ == '__main__':
    download_all_pairs(DATA)
