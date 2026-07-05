# Simple averaging fusion. This is Method 1 and is the actinng baseline. 
# Every output pixel is just the average of the two source pixels, so it's the simplest thing you can do. 
# It tends to wash out contrast , but it's a good base to compare the better methods against.
# All other methods should outperform this method for image fusion!

import numpy as np


def fuse_average(img_A, img_B):
    # add the two images and divide by 2 = the mean at every pixel.
    # clip to [0,1] just in case rounding pushes anything slightly out of range.
    return np.clip((img_A + img_B) / 2.0, 0.0, 1.0)
