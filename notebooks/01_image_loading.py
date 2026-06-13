# Remember to add startup snippet first (Finish snippet not necessary yet)

# Harvard Whole Brain Atlas Image Downloader
# Run once to add image pairs to your Google Drive.

# Downloads MRI+CT, MRI+SPECT, and MRI+PET brain image pairs
# Images are saved as PNG regardless of the original format on the server (some
# images are originally GIF files)

import urllib.request, os, numpy as np
# PIL allows us to work with GIF files from site
from PIL import Image as PILImage

BASE      = 'http://www.med.harvard.edu/AANLIB/cases'
CT_DIR    = f'{DATA}/mri_ct' # MRI+CT cases go here in Drive
SPECT_DIR = f'{DATA}/mri_spect' # MRI+SPECT cases go here in Drive
PET_DIR   = f'{DATA}/mri_pet' # MRI+PET cases go here in Drive
for d in [CT_DIR, SPECT_DIR, PET_DIR]:
    os.makedirs(d, exist_ok=True)

# Cases chosen based off how easily the case may be recognized by audience and
# highest number of valid files
# MRI+CT: anatomical + anatomical (bone/dense tissue vs soft tissue)
CT_CASES = [
    ('case13', 'stroke'),       # Acute stroke 
    ('case42', 'hemorrhage'),   # Cerebral hemorrhage 
    ('case41', 'toxoplasma'),   # Cerebral toxoplasmosis 
    ('case28', 'carcinoma'),    # Metastatic carcinoma 
]

# MRI+SPECT: anatomical + functional (blood flow)
SPECT_CASES = [
    ('case3',  'alzheimers'),   # Alzheimer's disease
    ('case11', 'huntingtons'),  # Huntington's disease
    ('case18', 'vascular_dem'), # Vascular dementia
    ('case25', 'encephalitis'), # Herpes encephalitis 
]

# MRI+PET: anatomical + functional (glucose metabolism)
# Note: caseNN1 uses PNG files on the server instead of GIF 
PET_CASES = [
    ('caseNN1', 'alzheimers'),  # Mild Alzheimer's disease + FDG-PET
    ('caseSLU', 'glioma'),      # Glioma + FDG-PET 
    ('caseSLU2','glioma2'),     # Second glioma case 
]

def download_image(case, prefix, s, save_path):
    # Try each folder variant (mr1/mr2/mr3/mr4) and both file formats (.gif and .png)
    # Some AANLIB cases store images as GIF, others as PNG — we try both
    # Whatever format downloads successfully gets converted to PNG and saved
    for n in ['1', '2', '3', '4']:
        for ext in ['.gif', '.png']:
            url = f'{BASE}/{case}/{prefix}{n}/{s}{ext}'
            try:
                # Download to a temporary file first so we can check quality
                tmp = save_path.replace('.png', f'_tmp{ext}')
                urllib.request.urlretrieve(url, tmp)

                # Check the image is not blank and is above a certain threshold
                arr = np.array(PILImage.open(tmp).convert('L'))
                # ADJUST MEAN VALUE IF YOU HAVE PROBLEMS
                if arr.mean() > 20:
                    # For valid images, resize to 256x256 if needed, then save as PNG
                    img = PILImage.open(tmp).convert('L')
                    if img.size != (256, 256):
                        print(f'    Resizing {case}/{prefix}{n}/{s} from {img.size} to (256, 256)')
                        img = img.resize((256, 256), PILImage.LANCZOS)
                    img.save(save_path)
                    os.remove(tmp)
                    return True

                # If mage was too dark, then delete and try new image
                os.remove(tmp)

            except:
                pass   # 404 or network error, try next image
    return False


def scan_and_download(cases, modality, folder, target_cases=2, pairs_per_case=5):
    # Scan images 015-060, as these tended to have the most image content
    # Only keep a case if we find the full target number of matched pairs
    # ADJUST RANGE IF HAVING ISSUES FINDING IMAGES
    # Figure out which modality label to use for print messages
    if modality == 'ct':
        label = 'CT'
    elif modality == 'tc':
        label = 'SPECT'
    else:
        label = 'PET'

    print(f'\nDownloading MRI+{label} pairs...')
    cases_done = 0

    for case_id, case_label in cases:
        if cases_done == target_cases:
            break
        pairs_found = 0
        saved_files = []

        for i in range(10, 61):
            if pairs_found == pairs_per_case:
                break
            s        = f'{i:03d}'
            # Save as PNG 
            mri_path = f'{folder}/{case_label}_mri_{s}.png'
            sec_path = f'{folder}/{case_label}_{modality}_{s}.png'

            got_mri = download_image(case_id, 'mr', s, mri_path)
            got_sec = download_image(case_id, modality, s, sec_path)

            if got_mri and got_sec:
                saved_files += [mri_path, sec_path]
                pairs_found += 1
                print(f'  {case_label} slice {s} — {pairs_found}/{pairs_per_case}')
            else:
                # One image is missing so discard both to avoid incomplete pairs
                for p in [mri_path, sec_path]:
                    if os.path.exists(p): os.remove(p)

        if pairs_found == pairs_per_case:
            print(f'  {case_label} complete — {pairs_found} pairs downloaded')
            cases_done += 1
        else:
            # Did not find enough pairs for this case so move on to the next one
            print(f'  {case_label} did not have enough pairs, trying next case')
            for p in saved_files:
                if os.path.exists(p): os.remove(p)


# This converts any existing GIFs in Drive to PNG so everything is consistent
existing_gifs = []
for folder in [CT_DIR, SPECT_DIR, PET_DIR]:
    existing_gifs += [f'{folder}/{f}' for f in os.listdir(folder) if f.endswith('.gif')]

if existing_gifs:
    print(f'Converting {len(existing_gifs)} existing GIF files to PNG...')
    for gif_path in existing_gifs:
        png_path = gif_path.replace('.gif', '.png')
        PILImage.open(gif_path).convert('L').save(png_path)
        os.remove(gif_path)
        print(f'  Converted: {os.path.basename(gif_path)}')
    print('Conversion complete.')


# Skip download if images are already in Drive. 
# Delete those images manually to try downloading new images
ct_n    = len(os.listdir(CT_DIR))
spect_n = len(os.listdir(SPECT_DIR))
pet_n   = len(os.listdir(PET_DIR))

if ct_n > 0 and spect_n > 0 and pet_n > 0:
    print('Images already downloaded:')
    print(f'  mri_ct/    {ct_n} files ({ct_n // 2} pairs)')
    print(f'  mri_spect/ {spect_n} files ({spect_n // 2} pairs)')
    print(f'  mri_pet/   {pet_n} files ({pet_n // 2} pairs)')
else:
    scan_and_download(CT_CASES,    'ct', CT_DIR)
    scan_and_download(SPECT_CASES, 'tc', SPECT_DIR)
    scan_and_download(PET_CASES,   'dg', PET_DIR)
    print('\nDownload complete:')
    for folder, lbl in [(CT_DIR, 'MRI+CT'), (SPECT_DIR, 'MRI+SPECT'), (PET_DIR, 'MRI+PET')]:
        n = len(os.listdir(folder))
        print(f'  {lbl}: {n} files ({n // 2} pairs)')
