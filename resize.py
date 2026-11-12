# Gets a second image onto the first image's grid, so the transforms can run on the pair.
# This only fixes the size. It does not line the images up

import numpy as np
from skimage.transform import resize


# Puts image onto a canvas of target_shape.
# keep_proportions scales it until it just fits and centres it on black,
# otherwise it is stretched to fill the canvas exactly.
def place_on_grid(image, target_shape, keep_proportions=True):
    image = np.asarray(image, dtype=np.float64)
    target_height, target_width = target_shape

    # nothing to do, and returning it untouched keeps same-size pairs exact
    if image.shape == tuple(target_shape):
        return image

    if not keep_proportions:
        stretched = resize(image, target_shape, order=1, anti_aliasing=True)
        return np.clip(stretched, 0.0, 1.0)

    height, width = image.shape
    scale = min(target_height / height, target_width / width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    # anti_aliasing blurs a little before shrinking, so fine detail does not turn into speckle
    scaled = resize(image, (new_height, new_width), order=1, anti_aliasing=True)

    canvas = np.zeros(target_shape, dtype=np.float64)
    top = (target_height - new_height) // 2
    left = (target_width - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = scaled

    return np.clip(canvas, 0.0, 1.0)


# pywt.swt2 refuses a size that is not a multiple of 2^levels, which is 4 here.
# Mirrors the bottom and right edges out to the next multiple.
def pad_to_multiple(image, multiple=4):
    height, width = image.shape
    extra_rows = -height % multiple
    extra_columns = -width % multiple

    if extra_rows == 0 and extra_columns == 0:
        return image

    return np.pad(image, ((0, extra_rows), (0, extra_columns)), mode='symmetric')


if __name__ == '__main__':
    tall = np.random.default_rng(0).random((300, 200))

    for keep in [True, False]:
        placed = place_on_grid(tall, (256, 256), keep_proportions=keep)
        filled_columns = int((placed.sum(axis=0) > 0).sum())
        print(f'  keep_proportions={keep}: shape {placed.shape}, columns with content {filled_columns}')

    same = place_on_grid(tall, (300, 200))
    print(f'  same size comes back untouched: {np.array_equal(same, tall)}')
    print(f'  padding (187, 250) gives {pad_to_multiple(np.zeros((187, 250))).shape}')