import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset import load_array


# =========================
# PATHS
# =========================
LR_DIR = r"C:\KLA\data\train\NoisyLR"
GT_DIRS = [
    r"C:\KLA\data\train\GT 0",
    r"C:\KLA\data\train\GT 1",
]
RESULT_DIR = r"C:\KLA\restored_demo_outputs"

OUT_DIR = r"C:\KLA\comparison_outputs"

# Number of comparisons to create
NUM_IMAGES = 50


# =========================
# HELPERS
# =========================
def find_gt(stem):
    """Find GT image with the same filename stem."""
    for gt_dir in GT_DIRS:
        path = os.path.join(gt_dir, stem + ".npy")
        if os.path.exists(path):
            return path

    return None


def normalize_for_display(arr):
    """
    Convert float array to displayable [0,1].
    Uses min-max normalization only for visualization.
    Does NOT modify the actual model output.
    """
    arr = np.asarray(arr).astype(np.float32)

    if arr.ndim == 3:
        arr = arr.squeeze()

    # For visualization, clip extreme values.
    arr = np.clip(arr, 0, 1)

    return arr


def array_to_image(arr):
    """Convert numpy array to PIL grayscale image."""
    arr = normalize_for_display(arr)

    arr = (arr * 255).round().astype(np.uint8)

    return Image.fromarray(arr, mode="L")


def make_comparison(stem):
    lr_path = os.path.join(LR_DIR, stem + ".npy")
    result_path = os.path.join(RESULT_DIR, stem + ".png")
    gt_path = find_gt(stem)

    if not os.path.exists(lr_path):
        print(f"Skipping {stem}: LR not found")
        return

    if not os.path.exists(result_path):
        print(f"Skipping {stem}: restored result not found")
        return

    if gt_path is None:
        print(f"Skipping {stem}: GT not found")
        return

    # Load arrays
    lr = load_array(lr_path)
    gt = load_array(gt_path)

    # Load model result
    result = np.array(Image.open(result_path).convert("L")).astype(np.float32) / 255.0

    # Convert to PIL
    lr_img = array_to_image(lr)
    result_img = array_to_image(result)
    gt_img = array_to_image(gt)

    # Resize LR to same display size as output/GT
    target_size = gt_img.size
    lr_img = lr_img.resize(target_size, Image.Resampling.BICUBIC)

    # Add labels
    label_height = 45
    gap = 10

    width = target_size[0]
    height = target_size[1]

    canvas_width = width * 3 + gap * 2
    canvas_height = height + label_height

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    draw = ImageDraw.Draw(canvas)

    # Paste images
    canvas.paste(lr_img.convert("RGB"), (0, label_height))
    canvas.paste(
        result_img.convert("RGB"),
        (width + gap, label_height)
    )
    canvas.paste(
        gt_img.convert("RGB"),
        (2 * (width + gap), label_height)
    )

    # Labels
    draw.text((10, 12), "Noisy LR", fill="black")
    draw.text((width + gap + 10, 12), "Our Restoration", fill="black")
    draw.text((2 * (width + gap) + 10, 12), "Ground Truth", fill="black")

    output_path = os.path.join(
        OUT_DIR,
        f"{stem}_comparison.png"
    )

    canvas.save(output_path)

    print(f"Created: {output_path}")


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(
        f for f in os.listdir(LR_DIR)
        if f.endswith(".npy")
    )

    if not files:
        print("No .npy files found.")
        return

    selected = files[:NUM_IMAGES]

    print("=" * 60)
    print("CREATING IMAGE COMPARISONS")
    print("=" * 60)

    for filename in selected:
        stem = os.path.splitext(filename)[0]
        make_comparison(stem)

    print("=" * 60)
    print("DONE")
    print(f"Output folder: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()