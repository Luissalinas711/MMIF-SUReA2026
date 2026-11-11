# The MMIF site. Run with: streamlit run app.py

import numpy as np
import streamlit as st
from PIL import Image as PILImage

st.title('MMIF Fusion Grid')

# Same loading as load_image_pair: grayscale, scaled to [0, 1]
def load_upload(uploaded_file):
    picture = PILImage.open(uploaded_file).convert('L')
    return np.array(picture, dtype=np.float32) / 255.0


left, right = st.columns(2)
file_a = left.file_uploader('Image A, the reference (MRI)', type=['png', 'jpg', 'gif'])
file_b = right.file_uploader('Image B (CT, PET or SPECT)', type=['png', 'jpg', 'gif'])

if file_a and file_b:
    image_a = load_upload(file_a)
    image_b = load_upload(file_b)

    left.image(image_a, caption=f'A: {image_a.shape}')
    right.image(image_b, caption=f'B: {image_b.shape}')
else:
    st.write('Choose both images to begin.')