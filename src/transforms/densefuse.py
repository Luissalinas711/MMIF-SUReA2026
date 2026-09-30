# Method 5: DenseFuse encoder

# difference from the old code: this calls the encoder and decoder directly 
# it also skips the network's own fusion step, b/c the fusion is my choice now 

# Encoder output comes out of a ReLU so it is never negative, which means maxabs becomes a maximum here.
# torch and the authors' net.py get imported inside the functions rather than at the top, so the other transforms still work without torch
# thus, the repo only has to be cloned when DenseFuse actually runs.

import os
import sys

import numpy as np

from coeffs import Decomposition

# Loaded models stay in memory, otherwise the weights get read again for every pair
MODEL_CACHE = {}


def torch_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def weights_file(weights_path=None):
    # Read from the environment every time rather than once at import
    if weights_path:
        return weights_path

    repo = os.environ.get('DENSEFUSE_REPO', 'densefuse-pytorch')
    return os.environ.get('DENSEFUSE_WEIGHTS',
                          os.path.join(repo, 'models', 'densefuse_gray.model'))


def load_model(weights_path):
    import torch

    if weights_path in MODEL_CACHE:
        return MODEL_CACHE[weights_path]

    # The authors' repo holds net.py next to the weights.
    # Appended rather than inserted so their utils.py does not block mine
    repo = os.environ.get('DENSEFUSE_REPO', 'densefuse-pytorch')
    if repo not in sys.path:
        sys.path.append(repo)
    from net import DenseFuse_net

    device = torch_device()
    model = DenseFuse_net(input_nc=1, output_nc=1)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)

    MODEL_CACHE[weights_path] = model
    return model


def to_tensor(image):
    # A [0,1] image becomes a (1, 1, height, width) tensor in [0,255], which is the range the model was trained on.
    import torch

    pixels = np.asarray(image, dtype=np.float32) * 255.0
    tensor = torch.from_numpy(pixels).float().unsqueeze(0).unsqueeze(0)

    return tensor.to(torch_device())


def to_image(tensor):
    # Back to a [0,1] numpy image.
    return tensor.clamp(0, 255).squeeze().cpu().numpy() / 255.0


def forward(image, weights_path=None):
    import torch

    weights = weights_file(weights_path)
    model = load_model(weights)

    with torch.no_grad():
        features = model.encoder(to_tensor(image))

    # encoder hands back a list holding one tensor shaped (1, channels, height, width)
    feature_maps = features[0].squeeze(0).cpu().numpy()

    return Decomposition(None, [feature_maps], {'weights': weights})


def inverse(decomposition):
    import torch

    model = load_model(decomposition.rebuild_info['weights'])

    features = torch.from_numpy(decomposition.detail_bands[0])
    features = features.float().unsqueeze(0).to(torch_device())

    with torch.no_grad():
        # decoder expects the same list shape the encoder produced
        rebuilt = model.decoder([features])[0]

    return to_image(rebuilt)
