# image loading, saving, and display utilities
import numpy as np, os, matplotlib.pyplot as plt
# PIL required because we are working with GIFs from Harvard Whole Brain Atlas
from PIL import Image as PILImage

# Load both images as grayscale arrays. Normalised to [0.0, 1.0]
def load_image_pair(path_A, path_B):
    def load(p):
        return np.array(PILImage.open(p).convert('L'), dtype=np.float32) / 255.0

    A, B = load(path_A), load(path_B)

    # All AANLIB images should be 256x256. Confirm this as a precautionary measure.
    assert A.shape == B.shape, f'Shape mismatch: {path_A} {A.shape} vs {path_B} {B.shape}'

    return A, B
  
# Convert float32 [0,1] back to uint8 [0,255]. Save as PNG
def save_fused(fused, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    PILImage.fromarray((fused * 255).astype(np.uint8)).save(path)

# Show images side by side
def display_comparison(*images, titles, save_path=None):
    fig, axes = plt.subplots(1, len(images), figsize=(4.5 * len(images), 4.5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
