# Method 5: DenseFuse encoder
# Reuses my fdensefuse.py adapter for the weights and the 0 to 1 versus 0 to 255 scaling.
# The difference is that this calls the encoder and decoder directly and skips the network's own fusion step

# torch gets imported inside the functions, not at the top, so the other four transforms still work without torch installed

import numpy as np

from coeffs import Decomposition


def forward(image, weights_path=None):
    import torch
    from fusion.densefuse import load_cached_model, to_tensor, DEFAULT_WEIGHTS

    weights = weights_path or DEFAULT_WEIGHTS
    model = load_cached_model(weights)

    with torch.no_grad():
        features = model.encoder(to_tensor(np.asarray(image, dtype=np.float32)))

    # encoder returns a list holding one tensor shaped (1, channels, height, width)
    feature_maps = features[0].squeeze(0).cpu().numpy()

    return Decomposition(None, [feature_maps], {'weights': weights})


def inverse(decomposition):
    import torch
    from fusion.densefuse import load_cached_model, to_numpy, DEVICE

    model = load_cached_model(decomposition.rebuild_info['weights'])

    features = torch.from_numpy(decomposition.detail_bands[0])
    features = features.float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        rebuilt = model.decoder([features])[0]

    return to_numpy(rebuilt)
