"""
Paired dataset loader for the KLA restoration challenge.

IMPORTANT — adapt this to your actual extracted folder structure:
The .rar archives were not inspected by the assistant (no unrar/network
in the build sandbox). This loader assumes, after you extract the
archives locally:

    data/GT_0/*.png  (or .npy/.tif)      <- ground truth, part 1
    data/GT_1/*.png  (or .npy/.tif)      <- ground truth, part 2 (rars
                                             were split for upload size,
                                             likely the same folder)
    data/NoisyLR/*.png (or .npy/.tif)    <- degraded 128x128 inputs

Pairing is done by matching filename stems (e.g. "0001.png" in
NoisyLR pairs with "0001.png" in GT_0 or GT_1). If your real filenames
don't share a stem 1:1, edit `_build_pairs()` below — that is the only
part of this file that encodes an assumption about naming.

Loading logic per format:
  - .npy        -> np.load, kept as float32, NO rescaling (you said the
                   arrays are already float32 with GT in [0,1] and
                   NoisyLR in [-0.279, 2.158], so we trust that as-is).
  - .png/.jpg/.tif (8-bit) -> loaded via PIL, divided by 255 -> [0,1].
    NOTE: if your NoisyLR files are stored as ordinary 8-bit images,
    they *cannot* natively hold values outside [0,1] or below 0 — in
    that case your NoisyLR is almost certainly stored as .npy or a
    float TIFF. Check the actual file extension after extraction and
    adjust `load_array()` if it's something else (e.g. 16-bit PNG).
"""
import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_array(path):
    """Load a single image/array file as float32 numpy, shape (H, W) or (H, W, C)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        return arr
    elif ext in IMG_EXTS:
        img = Image.open(path)
        arr = np.array(img).astype(np.float32)
        if arr.max() > 1.5:  # looks like 8-bit (or 16-bit) — normalize to [0,1]
            max_val = 65535.0 if arr.max() > 255 else 255.0
            arr = arr / max_val
        return arr
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


class RestorationDataset(Dataset):
    def __init__(self, gt_dirs, lr_dir, patch_size=None, augment=True,
                 extra_noise_aug=True, split="train", val_fraction=0.1, seed=42):
        """
        gt_dirs: list of directories containing ground-truth images
                 (e.g. ["data/GT_0", "data/GT_1"])
        lr_dir: directory containing degraded/noisy low-res images
        patch_size: if set, randomly crops GT to (patch_size*scale) and
                    matching LR patch — leave None to use full images
                    (your 128->256 pairs are already small, so full-image
                    training is fine and simpler).
        augment: random flips/rotations (safe, label-preserving).
        extra_noise_aug: on top of the real degradation already baked
                    into NoisyLR, randomly inject a bit of *additional*
                    synthetic Gaussian/speckle noise at train time. This
                    is the "synthetic data generation" the problem
                    statement explicitly recommends for out-of-distribution
                    robustness. Turn off if it hurts validation metrics.
        split: "train" or "val" — deterministic split via seed.
        """
        self.patch_size = patch_size
        self.augment = augment and split == "train"
        self.extra_noise_aug = extra_noise_aug and split == "train"

        gt_files = []
        for d in gt_dirs:
            for ext in IMG_EXTS | {".npy"}:
                gt_files += glob.glob(os.path.join(d, f"*{ext}"))
        lr_files = []
        for ext in IMG_EXTS | {".npy"}:
            lr_files += glob.glob(os.path.join(lr_dir, f"*{ext}"))

        gt_by_stem = {_stem(p): p for p in gt_files}
        lr_by_stem = {_stem(p): p for p in lr_files}
        common = sorted(set(gt_by_stem) & set(lr_by_stem))
        if not common:
            raise RuntimeError(
                "No matching filename stems found between GT and NoisyLR "
                "directories. Check your extracted folder structure and "
                "update dataset.py's pairing logic if filenames differ."
            )

        rng = random.Random(seed)
        rng.shuffle(common)
        n_val = max(1, int(len(common) * val_fraction))
        val_stems = set(common[:n_val])
        stems = [s for s in common if s in val_stems] if split == "val" \
            else [s for s in common if s not in val_stems]

        self.pairs = [(lr_by_stem[s], gt_by_stem[s]) for s in stems]
        print(f"[{split}] {len(self.pairs)} pairs "
              f"({len(common)} total matched, {n_val} held out for val)")

    def __len__(self):
        return len(self.pairs)

    def _to_chw_tensor(self, arr):
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    def __getitem__(self, idx):
        lr_path, gt_path = self.pairs[idx]
        lr = load_array(lr_path)
        gt = load_array(gt_path)

        if self.extra_noise_aug and random.random() < 0.5:
            if random.random() < 0.5:
                lr = lr + np.random.normal(0, random.uniform(0.005, 0.03), lr.shape).astype(np.float32)
            else:
                lr = lr + lr * np.random.normal(0, random.uniform(0.05, 0.15), lr.shape).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                lr, gt = np.fliplr(lr).copy(), np.fliplr(gt).copy()
            if random.random() < 0.5:
                lr, gt = np.flipud(lr).copy(), np.flipud(gt).copy()
            k = random.choice([0, 1, 2, 3])
            if k:
                lr, gt = np.rot90(lr, k).copy(), np.rot90(gt, k).copy()

        return self._to_chw_tensor(lr), self._to_chw_tensor(gt)
