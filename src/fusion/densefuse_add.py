import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Folder of the authors' cloned repo which holds net.py and the model weights
# Appended so their utils.py do not clash with ours
REPO_PATH = os.environ.get('DENSEFUSE_REPO', 'densefuse-pytorch')
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# Import the required coding architecture from the autors work
from net import DenseFuse_net

# Use the GPU for faster processing if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Keep loaded models in memory
MODEL_CACHE = {}


def load_model(weights_path):
    if weights_path not in MODEL_CACHE:
        model = DenseFuse_net(input_nc=1, output_nc=1)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        model.to(device)
        MODEL_CACHE[weights_path] = model
    return MODEL_CACHE[weights_path]


def to_tensor(image_array):

    # Scale [0,1] up to the [0,255] range the model was trained on
    image_array = image_array * 255.0

    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(image_array).float()

    # Add Batch and Channel dimensions so shape becomes (1, 1, H, W)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    return image_tensor.to(device)


def to_image(fused_tensor):

    # Constrain pixel values to strictly fall within the [0, 255] image range
    fused_tensor = fused_tensor.clamp(0, 255)

    # Remove the extra Batch/Channel dimensions and move the tensor back to the CPU
    fused_tensor = fused_tensor.squeeze().cpu()

    # Convert back to a standard NumPy array, scaled to [0, 1]
    return fused_tensor.numpy() / 255.0


def build_fuser(weights_path):

    def fuse(image_A, image_B):
        model = load_model(weights_path)

        # Load the images into tensors
        tensor_A = to_tensor(image_A)
        tensor_B = to_tensor(image_B)

        # Fusion needs both images the same size. If they differ, stretch the second one to match the first 
        #bilinear = weighted average of the 4 nearest pixels
        if tensor_B.shape != tensor_A.shape:
            tensor_B = F.interpolate(
                tensor_B,
                size=tensor_A.shape[2:],
                mode='bilinear',
                align_corners=False,
            )

        # Disable gradient calculation since we use a pre-trained DenseFuse model.
        # Not disabling this would lead to large unnecessary calculations.
        with torch.no_grad():

            # Encode both source images into deep feature maps
            features_A = model.encoder(tensor_A)
            features_B = model.encoder(tensor_B)

            # Fuse the feature maps using the addition strategy
            # Note: norm strategy commented out in original repository of author
            fused_features = model.fusion(
                features_A, features_B, strategy_type='addition')

            # Decode the fused feature map back into an image
            # The decoder returns a list, so take the first element 0
            fused_output = model.decoder(fused_features)[0]

        return to_image(fused_output)

    return fuse


# The default fuser uses the authors pre-trained weights
DEFAULT_WEIGHTS = os.environ.get(
    'DENSEFUSE_WEIGHTS', os.path.join(REPO_PATH, 'models', 'densefuse_gray.model'))
densefuse_fuse = build_fuser(DEFAULT_WEIGHTS)
