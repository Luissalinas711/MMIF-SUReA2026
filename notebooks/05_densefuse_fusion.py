# Method 4: DenseFuse Fusion
# Remember the MMIF Startup snippet first, then this cell, then MMIF Finish.
import os

DENSEFUSE_DIR = f'{REPO_PATH}/densefuse-pytorch'

# Get the authors repo. It comes with the network architecture and the pretrained weights
if not os.path.isfile(f'{DENSEFUSE_DIR}/net.py'):
    os.system(f'rm -rf {DENSEFUSE_DIR}')
    os.system(f'git clone https://github.com/hli1221/densefuse-pytorch {DENSEFUSE_DIR}')

# Tell the adapter where the authors repo is
os.environ['DENSEFUSE_REPO'] = DENSEFUSE_DIR     

from pipeline import fuse_all_pairs               
from fusion.densefuse_add import densefuse_fuse      


# Run DenseFuse over every pair, just like the other methods
fuse_all_pairs(
    densefuse_fuse,
    method='densefuse',
    data_root=DATA,
    results_root=f'{REPO_PATH}/results',
    label='DenseFuse Fused',
    show=True, # set false to not display images in coding environment
)
