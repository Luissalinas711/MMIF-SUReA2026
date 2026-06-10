#Remember to add startup snippet first (Finish snippet not necessary yet)

# Harvard Whole Brain Atlas Image Downloader
# Run once to add image pairs to your Google Drive.

import urllib.request
import os
import numpy as np
#PIL allows us to work with GIF files from site
from PIL import Image as PILImage

# DATA is set automatically by the startup snippet
CT_DIR  = f'{DATA}/mri_ct' # MRI + CT pairs go here
SPECT_DIR = f'{DATA}/mri_spect' # MRI + PET pairs go here
# Create the folders if they do not already exist
for d in [CT_DIR, SPECT_DIR]:
    os.makedirs(d, exist_ok=True)

BASE = 'http://www.med.harvard.edu/AANLIB/cases'

# Target Cases
# Confirmed working cases listed. First cases are priority (on preference)
# Scanner stops once 2 cases (5 pairs each) are secured per modality combination

CT_CASES = [
    'case13',   # Acute stroke
    'case42',   # Cerebral hemorrhage
    'case41',   # Cerebral toxoplasmosis
    'case28',   # Metastatic bronchogenic carcinoma
    'case10',   # Chronic subdural hematoma
    'case35',   # Hypertensive encephalopathy
]

SPECT_CASES = [
    'case3',    # Alzheimer's disease
    'case4',    # Stroke (subacute)
    'case11',   # Huntington's disease
    'case18',   # Vascular dementia
    'case25',   # Herpes encephalitis
    'case29',   # Alzheimer's disease (Second Alzheimer's case)
    'case15',   # Subacute stroke (second stroke case)
]

#Download Images
def download_image(case, prefix, s, path):
    # Try mr1/mr2/mr3 (MRI) or ct1/ct2/ct3 (CT)
    # or dg1/dg2 (PET) until a valid image is found.
    for n in ['1', '2', '3']:
        url = f'{BASE}/{case}/{prefix}{n}/{s}.gif'
        try:
            urllib.request.urlretrieve(url, path)
            arr = np.array(PILImage.open(path).convert('L'))
            if arr.max() > 0:
                return True      # valid non-blank image found
            os.remove(path)      # blank image found, try next image
        except:
            pass                 # 404 or network error, try again
    return False

# Scan and download matched pairs
def scan_and_download(case_list, modality, save_folder, n_cases=2, n_pairs=5):
    # Slices 015-060 (middle slices) yield best image quality
    # First/Last Images in each case had less/missing information on average
    label = 'CT' if modality == 'ct' else 'SPECT'
    print(f'MRI+{label} — targeting {n_cases} cases x {n_pairs} pairs each')

    cases_secured = 0

    for case in case_list:
        if cases_secured == n_cases:
            break

        pairs_found = 0
        saved_paths = []

        for i in range(15, 61):
            if pairs_found == n_pairs:
                break

            s        = f'{i:03d}'
            mri_path = f'{save_folder}/{case}_mri_{s}.gif'
            sec_path = f'{save_folder}/{case}_{modality}_{s}.gif'

            mri_ok = download_image(case, 'mr', s, mri_path)
            sec_ok = download_image(case, modality, s, sec_path)

            if mri_ok and sec_ok:
                # Both modalities valid, thus keep the pair
                saved_paths += [mri_path, sec_path]
                pairs_found += 1
                print(f'  [+] {case} slice {s} — pair {pairs_found}/{n_pairs}')
            else:
                # Only one modality found, thus discard both images
                for p in [mri_path, sec_path]:
                    if os.path.exists(p): os.remove(p)

        if pairs_found == n_pairs:
            print(f'  [OK] Secured: {case}')
            cases_secured += 1
        else:
            print(f'  [--] Skipped: {case} — only {pairs_found} pairs, purging')
            for p in saved_paths:
                if os.path.exists(p): os.remove(p)

# Checking number of files in each folder
ct_count    = len(os.listdir(CT_DIR))
spect_count = len(os.listdir(SPECT_DIR))

# Summarize the Results depending on 1 of 2 cases
# 1. Required number of files are already present. 
# 2. Files are successfully downloaded to Drive.
if ct_count > 0 and spect_count > 0:
    print('Images already present, skipping download.')
    print(f'  mri_ct/    : {ct_count} files  ({ct_count // 2} pairs)')
    print(f'  mri_spect/ : {spect_count} files  ({spect_count // 2} pairs)')
    print('Delete the files manually if you want to re-download.')
else:
    scan_and_download(CT_CASES,    'ct', CT_DIR)
    scan_and_download(SPECT_CASES, 'tc', SPECT_DIR)
    print()
    print('─' * 45)
    for d, label in [(CT_DIR, 'MRI+CT'), (SPECT_DIR, 'MRI+SPECT')]:
        n = len(os.listdir(d))
        print(f'{label}: {n} files  ({n // 2} pairs)')
