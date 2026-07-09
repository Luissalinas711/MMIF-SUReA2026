import os
import sys
import torch
import torch.nn.functional as F
import numpy as np


# Setup & Configuration
REPO_PATH = os.environ.get('DENSEFUSE_REPO', 'densefuse-pytorch')
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from net import DenseFuse_net 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Memory cache to prevent reloading the model on every function call
MODEL_CACHE = {}


def to_tensor(img_array):
    #Converts a [0, 1] NumPy array into a [0, 255] PyTorch tensor (1, 1, H, W)
    tensor = torch.from_numpy(img_array * 255.0).float()
    return tensor.unsqueeze(0).unsqueeze(0).to(DEVICE)

def to_numpy(tensor):
    #Converts a PyTorch tensor back into a normalized [0, 1] NumPy array
    image_array = tensor.clamp(0, 255).squeeze().cpu().numpy()
    return image_array / 255.0

def load_cached_model(weights_path):
    #Loads model weights into memory once and retrieves them on any other calls, as needed
    if weights_path not in MODEL_CACHE:
        model = DenseFuse_net(input_nc=1, output_nc=1)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        model.eval().to(DEVICE)
        MODEL_CACHE[weights_path] = model
    return MODEL_CACHE[weights_path]


# Main Fusion Stuff

def build_fuser(weights_path):
    
    def fuse(img_A, img_B):
        model = load_cached_model(weights_path)

        tensor_A = to_tensor(img_A)
        tensor_B = to_tensor(img_B)

        # Force spatial dimensions to match using bilinear interpolation if necessary
        # Bilinear interpolation calculates the value of a new pixel by taking a weighted average of the 4 closest original pixels. 
        # We use it here to smoothly stretch or shrink the secondary image so its spatial dimensions perfectly match the primary image.
        if tensor_B.shape != tensor_A.shape:
            tensor_B = F.interpolate(
                tensor_B, 
                size=tensor_A.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        # Execute Autoencoder Fusion
        with torch.no_grad():
            features_A = model.encoder(tensor_A)
            features_B = model.encoder(tensor_B)
            
            fused_features = model.fusion(features_A, features_B, strategy_type='addition')
            fused_tensor = model.decoder(fused_features)[0]

        return to_numpy(fused_tensor)

    return fuse

# The default pipeline uses the base pre-trained weights
DEFAULT_WEIGHTS = os.environ.get('DENSEFUSE_WEIGHTS', os.path.join(REPO_PATH, 'models', 'densefuse_gray.model'))
densefuse_fuse = build_fuser(DEFAULT_WEIGHTS)
