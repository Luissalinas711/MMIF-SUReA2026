# Shared fusion process.
# Every method notebook was running the exact same three loops (one per modality folder).
# Each notebook just imports its fusion method and runs it through fuse_all_pairs.

# For every pair we save two things:
#   results/<method>/fused/<name>.png     -> the fused image on its own
#   results/<method>/figures/<name>.png   -> the MRI , source , fused figure (this is purely for reference)
# Keeping them separate b/c the metrics step later needs the raw fused image

import os
from utils import load_image_pair, save_fused, display_comparison

# The two images in a pair sit in the same folder and only differ by a piece of the filename (e.g. alzheimers_mri_010.png and alzheimers_tc_010.png)
# To pair them I find the MRI file and swap the '_mri_' part for the partner's
# The '_mri_' part is the same for every folder, so it's just a constant
# The partner files are named by the TRACER, not the modality. SPECT files say 'tc' (technetium) and PET files say 'dg' (FDG), so I have to map each folder to that tracer token. 
# The display name also acts as the output suffix (just lowercased), so no separate column needed for that.
MRI_TOKEN = '_mri_'
MODALITIES = [
    # (subfolder,   partner token in the filename, display name)
    ('mri_ct',    '_ct_', 'CT'),
    ('mri_spect', '_tc_', 'SPECT'),    # tc = technetium tracer
    ('mri_pet',   '_dg_', 'PET'),      # dg = FDG tracer
]


# fusion_fn = whichever method we're running (takes mri + partner, return the fused image). 
# method = the output folder name ('averaging', 'dwt'...).
# label = title on the fused panel. 
# show = display each figure inline in Colab.
def fuse_all_pairs(fusion_fn, method, data_root, results_root,
                   label=None, show=True):
    # if no label was passed, just build one from the method name
    panel = label or f'{method.title()} Fused'
    total = 0   # running count of fused pairs

    # go through each modality folder one at a time
    for subfolder, partner_tok, modality in MODALITIES:
        folder = f'{data_root}/{subfolder}'
        # skip the folder if it isn't there, so one missing folder doesn't crash the whole run
        if not os.path.isdir(folder):
            print(f'  [skip] {folder} not found')
            continue

        # grab every MRI file (the partner file gets found by swapping the token down below)
        mri_files = sorted(f for f in os.listdir(folder)
                           if MRI_TOKEN in f and f.endswith('.png'))
        if not mri_files:
            print(f'  [skip] no MRI files in {subfolder} '
                  f'(run 01_download_images.py first)')
            continue

        suffix = modality.lower()   # 'CT' -> 'ct', gets tacked onto the output name

        # fuse every pair in this folder
        for mri_file in mri_files:
            # partner filename = same name but '_mri_' swapped for the tracer token
            partner_file = mri_file.replace(MRI_TOKEN, partner_tok)
            # output name: alzheimers_mri_010.png -> alzheimers_010 ...  (see below for rest)
            pair_name = mri_file.replace(MRI_TOKEN, '_').replace('.png', '')
            name = f'{pair_name}_{suffix}'   # ... then add on _ct / _spect / _pet

            # load both as grayscale [0,1], then run the fusion method
            mri, partner = load_image_pair(f'{folder}/{mri_file}',
                                           f'{folder}/{partner_file}')
            fused = fusion_fn(mri, partner)

            # save the fused image by itself (necessary for metric evaluation later)
            save_fused(fused, f'{results_root}/{method}/fused_only/{name}.png')

            # save the side-by-side figure too (again, purely for reference)
            display_comparison(
                mri, partner, fused,
                titles=['MRI', modality, panel],
                save_path=f'{results_root}/{method}/comparisons/{name}.png',
                show=show,
            )
            total += 1

        print(f'  {subfolder}: fused {len(mri_files)} pairs')

    # quick summary once everything is done
    print(f'[{method}] done. {total} fused images + {total} comparison images ')
    return total
