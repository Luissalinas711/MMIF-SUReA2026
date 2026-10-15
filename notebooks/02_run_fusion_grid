# Runs every transform and rule combination over the dataset. 15 combinations x 104 pairs = 1560 fused images
# Writes to Drive, not /content   

import os
from pipeline import run_all_combinations, check_averaging_column

# The startup cell clones DenseFuse at a pinned commit and sets DENSEFUSE_REPO
def densefuse_ready():
    repo = os.environ.get('DENSEFUSE_REPO', '')
    has_network = os.path.isfile(f'{repo}/net.py')
    has_weights = os.path.isfile(f'{repo}/models/densefuse_gray.model')

    print(f'densefuse repo at {repo or "DENSEFUSE_REPO not set"}: '
          f'network {has_network}, weights {has_weights}')
    return has_network and has_weights

if __name__ == '__main__':
    RESULTS = f'{DRIVE_ROOT}/results_grid'

    if not densefuse_ready():
        raise RuntimeError('run the startup cell first, the DenseFuse row needs it')
    run_all_combinations(DATA, RESULTS)

    # the four linear transforms must rebuild exactly under the average rule (another sanity check)
    # DATA not RESULTS, b/c it re-fuses a sample rather than reading the saved PNGs
    check_averaging_column(DATA)
