# Image loading, saving, and display utilities
import numpy as np, os, matplotlib.pyplot as plt
# we converted from gif to png, but PIL still works fine here.
from PIL import Image as PILImage

# Check that every image in the folder is the expected size.
def verify_image_sizes(folder, expected=(256, 256)):
    files  = sorted(f for f in os.listdir(folder) if f.endswith('.png'))  # change .gif to .png
    issues = []
    for f in files:
        shape = np.array(PILImage.open(f'{folder}/{f}').convert('L')).shape
        if shape != expected:
            issues.append(f'  {f}: got {shape}, expected {expected}')
    if issues:
        print(f'{os.path.basename(folder)}: size mismatch found!')
        for msg in issues: print(msg)
    else:
        print(f'{os.path.basename(folder)}: all {len(files)} images are {expected}')
    return len(issues) == 0

# Load both images as grayscale arrays. Normalised to [0.0, 1.0]
def load_image_pair(path_A, path_B):
    # PIL handles greyscale conversion
    def load(p):
        return np.array(PILImage.open(p).convert('L'), dtype=np.float32) / 255.0
    A, B = load(path_A), load(path_B)
    # Both images must be the same size in order for fusion methods to work
    if A.shape != B.shape:
        raise ValueError(f'Shape mismatch: {path_A} {A.shape} vs {path_B} {B.shape}')
    return A, B

def save_fused(fused, path):
    # Convert [0,1] back to [0,255] scale. Save as PNG when done
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PILImage.fromarray((fused * 255).astype(np.uint8)).save(path)

# Show (and/or save) images side by side
def display_comparison(*images, titles, save_path=None, show=True):
    fig, axes = plt.subplots(1, len(images), figsize=(4.5 * len(images), 4.5))
    axes = np.atleast_1d(axes)   # so a single-image call still works
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)          # avoids buildup over a batch
