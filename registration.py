# Lines the second image up with the first once both are on the same grid.
# Searches over shift, rotation and uniform zoom for the position where the two images
# share the most information. It cannot undo warping or fix a pair from different slices.

import numpy as np
from scipy.ndimage import affine_transform
from skimage.transform import resize

# How far the search is allowed to go. Shifts are a fraction of the image size.
MAX_SHIFT_FRACTION = 0.30
MAX_ANGLE_DEGREES = 25.0
MIN_SCALE, MAX_SCALE = 0.70, 1.40

# The search runs on small copies first and works up to larger ones.
LEVEL_SIZES = [48, 96, 192]


# (H(A) + H(B)) / H(A, B). 1 means the images tell you nothing about each other, 2 means identical.
# It never asks that bright match bright, only that one image predicts the other,
# which is why it works between MRI and CT where bone is dark in one and bright in the other.
def normalized_mutual_information(image_a, image_b, bins=32):
    joint_counts, _, _ = np.histogram2d(image_a.ravel(), image_b.ravel(),
                                        bins=bins, range=[[0, 1], [0, 1]])
    joint = joint_counts / joint_counts.sum()

    def entropy(probabilities):
        nonzero = probabilities[probabilities > 0]
        return -np.sum(nonzero * np.log(nonzero))

    joint_entropy = entropy(joint)
    if joint_entropy == 0:
        return 1.0

    return (entropy(joint.sum(axis=1)) + entropy(joint.sum(axis=0))) / joint_entropy


# Shifts, turns and zooms an image about its centre. Whatever slides in from outside is black.
# shift_x is to the right and shift_y is down, both in pixels. A positive angle is clockwise.
def move_image(image, shift_x=0.0, shift_y=0.0, angle_degrees=0.0, scale=1.0):
    angle = np.deg2rad(angle_degrees)
    cos, sin = np.cos(angle), np.sin(angle)

    # affine_transform works backwards: for each output pixel it asks where in the input to look.
    # So this is the inverse move, written for (row, column) order.
    backwards = np.array([[cos, -sin],
                          [sin, cos]]) / scale

    height, width = image.shape
    centre = np.array([(height - 1) / 2, (width - 1) / 2])
    offset = centre - backwards @ (centre + np.array([shift_y, shift_x]))

    return affine_transform(image, backwards, offset=offset, order=1, mode='constant', cval=0.0)


def inside_limits(move):
    return (abs(move['shift_x']) <= MAX_SHIFT_FRACTION and abs(move['shift_y']) <= MAX_SHIFT_FRACTION
            and abs(move['angle_degrees']) <= MAX_ANGLE_DEGREES
            and MIN_SCALE <= move['scale'] <= MAX_SCALE)


# Finds the move that best lines moving up with reference. Both must already be the same shape.
# Returns the move in pixels, plus the score before and after so the caller can judge it.
def register(reference, moving):
    height, width = reference.shape
    longest = max(height, width)
    level_sizes = [size for index, size in enumerate(LEVEL_SIZES) if index == 0 or size <= longest]

    # shifts are kept as a fraction of the image here, so one move means the same thing at every level
    best = {'shift_x': 0.0, 'shift_y': 0.0, 'angle_degrees': 0.0, 'scale': 1.0}
    best_score = -np.inf

    for level_index, level_size in enumerate(level_sizes):
        factor = min(1.0, level_size / longest)
        level_shape = (max(8, round(height * factor)), max(8, round(width * factor)))
        small_reference = resize(reference, level_shape, order=1, anti_aliasing=True)
        small_moving = resize(moving, level_shape, order=1, anti_aliasing=True)
        bins = 16 if level_index == 0 else 32   # few pixels cannot fill many bins

        def score(move):
            moved = move_image(small_moving, move['shift_x'] * level_shape[1], move['shift_y'] * level_shape[0],
                               move['angle_degrees'], move['scale'])
            return normalized_mutual_information(small_reference, moved, bins)

        if level_index == 0:
            # a coarse sweep over everything, so the search cannot get stuck near the start
            best_score = score(best)
            reach = round(max(level_shape) * 0.25)

            for angle in [-12, -6, 0, 6, 12]:
                for scale in [0.85, 0.92, 1.0, 1.08, 1.17]:
                    for pixels_y in range(-reach, reach + 1, 2):
                        for pixels_x in range(-reach, reach + 1, 2):
                            candidate = {'shift_x': pixels_x / level_shape[1], 'shift_y': pixels_y / level_shape[0],
                                         'angle_degrees': float(angle), 'scale': scale}
                            candidate_score = score(candidate)
                            if candidate_score > best_score:
                                best_score, best = candidate_score, candidate
        else:
            best_score = score(best)

        # then fine tuning: nudge one parameter at a time, keep what helps, halve the nudge when nothing does
        steps = {'shift_x': 2 / level_shape[1], 'shift_y': 2 / level_shape[0], 'angle_degrees': 2.0, 'scale': 0.03}

        for _ in range(40):
            if steps['shift_x'] * level_shape[1] <= 0.2:
                break

            improved = False
            for name in steps:
                for direction in [-1, 1]:
                    trial = dict(best)
                    trial[name] += direction * steps[name]
                    if not inside_limits(trial):
                        continue

                    trial_score = score(trial)
                    if trial_score > best_score:
                        best_score, best, improved = trial_score, trial, True

            if not improved:
                steps = {name: step / 2 for name, step in steps.items()}

    move = {'shift_x': best['shift_x'] * width, 'shift_y': best['shift_y'] * height,
            'angle_degrees': best['angle_degrees'], 'scale': best['scale']}

    # both scores on the full size images, so the two numbers compare
    before = normalized_mutual_information(reference, moving)
    after = normalized_mutual_information(reference, move_image(moving, **move))

    return move, before, after


# True when the move is big enough to be real and the score clearly got better.
# Without this the search would nudge pairs that were already lined up.
def worth_applying(move, before, after):
    is_real_move = (abs(move['shift_x']) > 0.25 or abs(move['shift_y']) > 0.25
                    or abs(move['angle_degrees']) > 0.25 or abs(move['scale'] - 1) > 0.004)
    return is_real_move and after > before * 1.002


# Knock an image out of place by a known amount, then see whether register puts it back.
if __name__ == '__main__':
    import time

    rows, columns = np.mgrid[0:256, 0:256]

    def ellipse(centre_y, centre_x, radius_y, radius_x):
        return (((rows - centre_y) / radius_y) ** 2 + ((columns - centre_x) / radius_x) ** 2) < 1

    # a stand in for an MRI: dark skull ring, grey tissue, bright blobs
    mri_like = (0.4 * ellipse(128, 128, 115, 100) - 0.35 * ellipse(128, 128, 105, 92)
                + 0.5 * ellipse(128, 128, 95, 84) + 0.2 * ellipse(110, 95, 50, 28)
                + 0.2 * ellipse(110, 160, 50, 28) + 0.35 * ellipse(90, 170, 14, 12))
    # and for a CT of the same head: bright skull ring, flat tissue
    ct_like = (0.2 * ellipse(128, 128, 115, 100) + 0.75 * ellipse(128, 128, 105, 92)
               - 0.65 * ellipse(128, 128, 95, 84) + 0.1 * ellipse(90, 170, 14, 12))

    knocked = move_image(ct_like, shift_x=9, shift_y=-6, angle_degrees=5, scale=0.93)

    started = time.time()
    move, before, after = register(mri_like, knocked)
    restored = move_image(knocked, **move)

    print(f'  found: right {move["shift_x"]:.1f} px, down {move["shift_y"]:.1f} px, '
          f'{move["angle_degrees"]:.1f} degrees, scale {move["scale"]:.3f}')
    print(f'  expected roughly the reverse of: right 9, down -6, 5 degrees, scale 0.93')
    print(f'  mutual information {before:.3f} -> {after:.3f}')
    print(f'  mean error against the true image {np.abs(knocked - ct_like).mean():.4f} '
          f'-> {np.abs(restored - ct_like).mean():.4f}')
    print(f'  worth applying: {worth_applying(move, before, after)}, took {time.time() - started:.1f} s')

    move, before, after = register(mri_like, ct_like)
    print(f'  already aligned pair, worth applying: {worth_applying(move, before, after)}')