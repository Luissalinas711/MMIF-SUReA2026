# Method 2: Laplacian pyramid fusion.
# Remember the MMIF Startup snippet first, then this cell, then MMIF Finish.

from fusion.laplacian import laplacian_pyramid_fuse
from pipeline import fuse_all_pairs

fuse_all_pairs(
    laplacian_pyramid_fuse,
    method='laplacian',
    data_root=DATA,
    results_root=f'{REPO_PATH}/results',
    label='Laplacian Fused',
    show=True,          # set False to just save w/out showing every figure
)
