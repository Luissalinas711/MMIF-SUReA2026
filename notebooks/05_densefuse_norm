# Method 4, norm strategy 
# Fuse the same pairs with the selective norm rule of DenseFuse instead of the addition rule
# Remember the MMIF Startup snippet first, then this cell, then MMIF Finish.

import os

# DenseFuse needs the authors' repo (for the pretrained model). 
if not os.path.isfile('densefuse-pytorch/net.py'):
    os.system('rm -rf densefuse-pytorch')
    os.system('git clone https://github.com/hli1221/densefuse-pytorch')
os.environ['DENSEFUSE_REPO'] = 'densefuse-pytorch'

from fusion.densefuse_norm import densefuse_fuse   # the norm-strategy fuser
from pipeline import fuse_all_pairs

fuse_all_pairs(
    densefuse_fuse,
    method='densefuse_norm',
    data_root=DATA,
    results_root=f'{REPO_PATH}/results',
    label='DenseFuse (norm)',
    show=True,
)
