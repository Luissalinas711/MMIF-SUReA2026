# Method 3: Discrete Wavelet Transform (DWT) fusion.
# Remember the MMIF Startup snippet first, then this cell, then MMIF Finish.

from fusion.dwt import dwt_fuse
from pipeline import fuse_all_pairs

fuse_all_pairs(
    dwt_fuse,
    method='dwt',
    data_root=DATA,
    results_root=f'{REPO_PATH}/results',
    label='DWT Fused',
    show=True,          # set False to just save w/out showing every figure
)
