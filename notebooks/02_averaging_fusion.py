# Method 1: Averaging Fusion.
# Remember to run the MMIF Startup snippet first (it sets DATA, REPO_PATH and puts src/ on the path). 
# Then run this cell and MMIF Finish at the end.

from fusion.averaging import fuse_average
from pipeline import fuse_all_pairs

# run averaging on every pair and save the results
fuse_all_pairs(
    fuse_average,
    method='averaging',
    data_root=DATA,
    results_root=f'{REPO_PATH}/results',
    label='Average Fused',
    show=True,          # set False to save w/out showing every figure
)
