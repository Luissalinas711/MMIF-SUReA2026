# The MMIF site. Run with: streamlit run app.py

import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image as PILImage

# The files in src/ import each other by bare name, like 'from coeffs import fuse',
# so src/ has to be on the path. The Colab startup cell does the same thing.
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from coeffs import fuse
from rules import RULES
from transforms import TRANSFORMS, CLASSICAL_TRANSFORMS
from resize import place_on_grid, pad_to_multiple
from registration import register, move_image, worth_applying
from metrics import all_metrics

# DenseFuse lives outside this repo: the authors' network file and their pretrained weights.
# src/transforms/densefuse.py finds them through this environment variable.
DENSEFUSE_REPO = Path(__file__).parent / 'densefuse-pytorch'
os.environ['DENSEFUSE_REPO'] = str(DENSEFUSE_REPO)

# The same commit colab_setup.py pins, so every copy of the site runs the same model
DENSEFUSE_COMMIT = '4394b63e9295db1c6b7a5c3664551c90f0605f2b'
DENSEFUSE_SOURCE = f'https://raw.githubusercontent.com/hli1221/densefuse-pytorch/{DENSEFUSE_COMMIT}'
DENSEFUSE_FILES = ['net.py', 'fusion_strategy.py', 'models/densefuse_gray.model']

# Every pair is brought to this size before fusion. It is the size of the study's dataset.
WORKING_SHAPE = (256, 256)

# Left to right in the order the rules commit to one source
RULE_ORDER = ['average', 'l1norm', 'maxabs']

# The keys all_metrics returns, with the names shown on the page
METRIC_LABELS = {
    'entropy': 'Entropy',
    'MI': 'Mutual information',
    'std': 'Standard deviation',
    'spatial_freq': 'Spatial frequency',
    'SSIM_mri': 'SSIM to A',
    'SSIM_partner': 'SSIM to B',
    'edge_preservation': 'Edge preservation',
}

# What each metric measures and how to read it, shown as a guide above the table
METRIC_NOTES = {
    'entropy': ('How much information the fused image holds, from the spread of its grey levels.',
                'Higher means more. Usually better, but noise and artifacts raise it too.'),
    'MI': ('How much of the two source images survives in the fused one, added over both sources.',
           'Higher is better.'),
    'std': ('Contrast: how widely the grey levels are spread around their mean.',
            'Higher means more contrast. Usually better, but not proof of quality by itself.'),
    'spatial_freq': ('Sharpness and fine detail, from changes between neighbouring pixels.',
                     'Higher means sharper. Ringing around edges also raises it.'),
    'SSIM_mri': ('How closely the structure of the fused image matches image A. 1 is identical.',
                 'Higher means closer to A. It tends to fall as SSIM to B rises.'),
    'SSIM_partner': ('How closely the structure of the fused image matches image B. 1 is identical.',
                     'Higher means closer to B. It tends to fall as SSIM to A rises.'),
    'edge_preservation': ('The share of the sources\' edges that survive in the fused image, about 0 to 1.',
                          'Higher is better.'),
}

# Where the study's images came from. The site links there and never copies or stores its images.
ATLAS_URL = 'https://www.med.harvard.edu/AANLIB/home.html'

# The Harvard server is sometimes unreachable. This is a copy of the same atlas on another university's server.
ATLAS_MIRROR_URL = 'https://www3.med.unipmn.it/did/will/BrainAtlas/home.html'

# The winning cells in the results table get this background, with white text
WINNER_COLOUR = '#2a78d6'

st.title('MMIF Fusion Grid')

st.write('Upload two brain scans of the same slice, one MRI and one CT, PET or SPECT. '
         'The page fuses them with every transform and rule from the MMIF study and scores each result.')

st.markdown(f'Need a pair to try? The study used images from [The Whole Brain Atlas]({ATLAS_URL}) '
            f'by Keith A. Johnson and J. Alex Becker, Harvard Medical School. '
            f'If the Harvard site is not responding, the same atlas can be reached through '
            f'[a copy hosted at unipmn.it]({ATLAS_MIRROR_URL}).')

with st.expander('How to pick an image pair from the atlas'):
    st.markdown(
        '1. Open the atlas and choose a case from one of its sections, such as Neoplastic Disease. '
        'Not every case has a second kind of scan, so look for one that offers CT, SPECT or PET as well as MRI.\n'
        '2. Pick one slice and view it as MRI, then view the same slice number in the second kind of scan. '
        'Slices from the middle of the head work best, since the top and bottom ones are mostly empty.\n'
        '3. Save both images to your computer.\n'
        '4. Come back here and upload the MRI as image A and the other scan as image B.'
    )
    st.caption('The atlas images belong to their authors. The atlas has a Copyright Information page '
               'that says what they may be used for, so read it before reusing them anywhere else.')

st.caption('Uploaded images are used to produce the results on this page and are not saved. '
           'Do not upload scans that could identify a patient. '
           'This is an independent student research project. It is not affiliated with Harvard Medical School '
           'or the atlas authors, and it is not for clinical use.')


# Same loading as load_image_pair: grayscale, scaled to [0, 1]
def load_upload(uploaded_file):
    picture = PILImage.open(uploaded_file).convert('L')
    return np.array(picture, dtype=np.float32) / 255.0


# A hosted copy of the site starts with no densefuse-pytorch folder, since it is not committed here.
# This fetches the three files it needs from the authors' own repository, once.
# On your own computer the folder is already there, so nothing is downloaded.
@st.cache_resource(show_spinner='Getting the DenseFuse model')
def fetch_densefuse():
    try:
        for name in DENSEFUSE_FILES:
            target = DENSEFUSE_REPO / name
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f'{DENSEFUSE_SOURCE}/{name}', target)
        return True
    except OSError:
        return False


# The same test as notebook 02, plus a check that torch is installed
def densefuse_ready():
    try:
        import torch
    except ImportError:
        return False

    if not fetch_densefuse():
        return False

    has_network = (DENSEFUSE_REPO / 'net.py').is_file()
    has_weights = (DENSEFUSE_REPO / 'models' / 'densefuse_gray.model').is_file()
    return has_network and has_weights


# Streamlit reruns this whole file on every click. The cache keeps a fused image
# for inputs it has already seen, so only new work gets done.
@st.cache_data(show_spinner=False)
def fuse_cell(image_a, image_b, transform_name, rule_name):
    transform, rule = TRANSFORMS[transform_name], RULES[rule_name]

    # the SWT is the one transform that is picky about size, so pad for it and trim after
    if transform_name == 'swt':
        height, width = image_a.shape
        fused = fuse(pad_to_multiple(image_a), pad_to_multiple(image_b), transform, rule)
        return fused[:height, :width]

    return fuse(image_a, image_b, transform, rule)


# The study scored fused images after save_fused wrote them as 8 bit PNGs and
# load_grayscale read them back. This does the same rounding without touching the disk,
# so the numbers here are measured the way the ones in metrics_long.csv were.
def as_saved(fused):
    return (fused * 255).astype(np.uint8).astype(np.float32) / 255.0


@st.cache_data(show_spinner=False)
def score_cell(image_a, image_b, fused):
    return all_metrics(image_a, image_b, as_saved(fused))


# One row per cell, one column per metric, plus a count of how many metrics that cell wins
def scores_table(all_scores):
    rows = {f'{transform_name}__{rule_name}': {METRIC_LABELS[key]: value for key, value in scores.items()}
            for (transform_name, rule_name), scores in all_scores.items()}
    table = pd.DataFrame.from_dict(rows, orient='index')

    table['Metrics won'] = find_winners(table).sum(axis=1)
    return table


# True wherever a cell has the highest value in its column.
# Values are compared as shown, to four decimals, so cells that print the same number tie.
# That matters for the average rule, where four transforms give the same image.
def find_winners(table):
    shown = table[list(METRIC_LABELS.values())].round(4)
    return shown.eq(shown.max(axis=0), axis=1)


# Colours the winning cells and formats the numbers
def style_table(table):
    winners = find_winners(table)

    def colour_winners(column):
        if column.name not in winners:
            return [''] * len(column)
        return [f'background-color: {WINNER_COLOUR}; color: white; font-weight: bold' if won else ''
                for won in winners[column.name]]

    number_format = {label: '{:.4f}' for label in METRIC_LABELS.values()}
    return table.style.apply(colour_winners, axis=0).format(number_format)


# The cell or cells that win the most metrics, as a sentence
def overall_summary(table):
    most = int(table['Metrics won'].max())
    leaders = list(table.index[table['Metrics won'] == most])
    total = len(METRIC_LABELS)

    if len(leaders) == 1:
        return f'{leaders[0]} wins the most metrics on this pair, {most} of {total}.'
    return f'{", ".join(leaders)} share the lead on this pair, each winning {most} of {total} metrics.'


# The same numbers in the layout of metrics_long.csv: one row per measurement
def scores_long_csv(all_scores):
    rows = [{'transform': transform_name, 'rule': rule_name, 'metric': key, 'value': value}
            for (transform_name, rule_name), scores in all_scores.items()
            for key, value in scores.items()]
    return pd.DataFrame(rows).to_csv(index=False)


# The guide as a small table: metric, what it measures, how to read it
def metric_guide():
    lines = ['| Metric | What it measures | How to read it |', '|---|---|---|']
    for key, label in METRIC_LABELS.items():
        what, reading = METRIC_NOTES[key]
        lines.append(f'| {label} | {what} | {reading} |')
    return '\n'.join(lines)


# The search takes a second or two, so it is cached like the fusions
@st.cache_data(show_spinner=False)
def find_alignment(reference, moving):
    return register(reference, moving)


# numpy says (rows, columns), people say width by height
def size_text(shape):
    return f'{shape[1]} \u00d7 {shape[0]}'


left, right = st.columns(2)
file_a = left.file_uploader('Image A, the reference (MRI)', type=['png', 'jpg', 'gif'])
file_b = right.file_uploader('Image B (CT, PET or SPECT)', type=['png', 'jpg', 'gif'])

if not (file_a and file_b):
    st.write('Choose both images to begin.')
    st.stop()

source_a = load_upload(file_a)
source_b = load_upload(file_b)

# Both images go onto the same 256 by 256 grid, with their proportions kept
image_a = place_on_grid(source_a, WORKING_SHAPE)
resized_b = place_on_grid(source_b, WORKING_SHAPE)

# Then B is lined up with A. The move is only used when it clearly helps,
# so pairs that are already registered pass through untouched.
with st.spinner('Preparing the images'):
    move, score_before, score_after = find_alignment(image_a, resized_b)

was_aligned = worth_applying(move, score_before, score_after)
image_b = move_image(resized_b, **move) if was_aligned else resized_b

# The one thing the user is told: what came in, and what the fusion ran on
left.caption(f'Image A: {size_text(source_a.shape)}')
right.caption(f'Image B: {size_text(source_b.shape)}')

if source_a.shape == WORKING_SHAPE and source_b.shape == WORKING_SHAPE and not was_aligned:
    st.write(f'Both images are already {size_text(WORKING_SHAPE)}, so they were fused as they are.')
else:
    st.write(f'The images were adjusted to fit {size_text(WORKING_SHAPE)} before fusion.')

# Everything else stays out of the way unless someone asks for it
with st.expander('How the images were prepared'):
    if was_aligned:
        st.write(f'Image B was moved {move["shift_x"]:.1f} px across and {move["shift_y"]:.1f} px down, '
                 f'turned {move["angle_degrees"]:.1f} degrees and scaled by {move["scale"]:.3f} to line up with image A.')
    else:
        st.write('Image B already lined up with image A, so it was not moved.')

    # A in green, B in red and blue. Where the two agree the overlay turns grey.
    overlay = np.stack([image_b, image_a, image_b], axis=-1)

    check_a, check_b, check_overlay = st.columns(3)
    check_a.image(image_a, caption='A as fused')
    check_b.image(image_b, caption='B as fused')
    check_overlay.image(overlay, caption='A green, B magenta')

st.header('Fused results')

transform_names = list(CLASSICAL_TRANSFORMS)
if densefuse_ready():
    transform_names.append('densefuse')
else:
    st.info('The DenseFuse row is off. It needs torch installed and the DenseFuse model files, which could not be found or downloaded.')

# One row per transform, one column per rule. The scores are kept for the table and charts below.
all_scores = {}

for transform_name in transform_names:
    st.subheader(transform_name)
    row = st.columns(len(RULE_ORDER))

    for column, rule_name in zip(row, RULE_ORDER):
        with st.spinner(f'{transform_name} with {rule_name}'):
            fused = fuse_cell(image_a, image_b, transform_name, rule_name)
            all_scores[(transform_name, rule_name)] = score_cell(image_a, image_b, fused)

        column.image(fused, caption=f'{transform_name}__{rule_name}')

st.header('How the cells compare')

st.markdown(metric_guide())

# One table: every score, with the highest value in each column picked out
st.subheader('Results')
table = scores_table(all_scores)
st.dataframe(style_table(table))

st.write(f'The highlighted value in each column is the highest for that metric. {overall_summary(table)}')
st.caption('Winning the most metrics is not the same as being the best fusion. The metrics reward different things, '
           'so a cell that leads on detail often trails on similarity to the sources. '
           'These numbers also describe one pair of images, while the study\'s conclusions come from 104 pairs.')

st.download_button('Download these numbers as CSV', scores_long_csv(all_scores),
                   file_name='mmif_metrics.csv', mime='text/csv')