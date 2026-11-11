# The MMIF site. Run with: streamlit run app.py

import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image as PILImage

# The files in src/ import each other by bare name, like 'from coeffs import fuse',
# so src/ has to be on the path. The Colab startup cell does the same thing.
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from coeffs import fuse
from rules import RULES
from transforms import TRANSFORMS, CLASSICAL_TRANSFORMS

# Left to right in the order the rules commit to one source
RULE_ORDER = ['average', 'l1norm', 'maxabs']

st.title('MMIF Fusion Grid')


# Same loading as load_image_pair: grayscale, scaled to [0, 1]
def load_upload(uploaded_file):
    picture = PILImage.open(uploaded_file).convert('L')
    return np.array(picture, dtype=np.float32) / 255.0


left, right = st.columns(2)
file_a = left.file_uploader('Image A, the reference (MRI)', type=['png', 'jpg', 'gif'])
file_b = right.file_uploader('Image B (CT, PET or SPECT)', type=['png', 'jpg', 'gif'])

if not (file_a and file_b):
    st.write('Choose both images to begin.')
    st.stop()

image_a = load_upload(file_a)
image_b = load_upload(file_b)

left.image(image_a, caption=f'A: {image_a.shape}')
right.image(image_b, caption=f'B: {image_b.shape}')

# The transforms need both images on the same grid. Resizing comes in the next step.
if image_a.shape != image_b.shape:
    st.error(f'The two images are different sizes, {image_a.shape} and {image_b.shape}. '
             f'For now they have to match.')
    st.stop()

st.header('Fused results')

# One row per transform, one column per rule
for transform_name in CLASSICAL_TRANSFORMS:
    st.subheader(transform_name)
    row = st.columns(len(RULE_ORDER))

    for column, rule_name in zip(row, RULE_ORDER):
        fused = fuse(image_a, image_b, TRANSFORMS[transform_name], RULES[rule_name])
        column.image(fused, caption=f'{transform_name}__{rule_name}')