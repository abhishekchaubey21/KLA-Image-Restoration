"""
Generates what you need for Slide 6 (Results): SSIM/pSNR/LPIPS on your
held-out val split, plus side-by-side degraded/restored/ground-truth
comparison images.

Usage:
    python make_results_report.py --gt_dirs data/GT_0 data/GT_1 \
        --lr_dir data/NoisyLR --checkpoint checkpoints/best_model.pt \
        --out_dir results
"""
import argparse
import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset import RestorationDataset
from model import build_model
from metrics import evaluate


def make_comparison_grid(lr, pred, gt, out_path):
    """lr: low-res input (smaller), pred/gt: same size. Upsamples lr with
    nearest-neighbor just for side-by-side display (not used in scoring)."""
    def to_uint8(t):
        a = t.clamp(0, 1).cpu().numpy()
        a = np.transpose(a, (1, 2, 0))
        if a.shape[2] == 1:
            a = a[:, :, 0]
        return (a * 255).astype(np.uint8)

    h, w = pred.shape[-2:]
    lr_img = Image.fromarray(to_uint8(lr)).resize((w, h), Image.NEAREST)
    pred_img = Image.fromarray(to_uint8(pred))
    gt_img = Image.fromarray(to_uint8(gt))

    mode = "L" if pred.shape[0] == 1 else "RGB"
    grid = Image.new(mode, (w * 3 + 20, h), color=255)
    grid.paste(lr_img, (0, 0))
    grid.paste(pred_img, (w + 10, 0))
    grid.paste(gt_img, (2 * w + 20, 0))
    grid.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt_dirs", nargs="+", required=True)
    p.add_argument("--lr_dir", required=True)
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--n_examples", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model(in_nc=ckpt["in_nc"], out_nc=ckpt["in_nc"],
                         scale=ckpt["scale"], size=ckpt["model_size"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    val_ds = RestorationDataset(args.gt_dirs, args.lr_dir, split="val",
                                 augment=False, extra_noise_aug=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    metrics = evaluate(model, val_loader, device, in_nc=ckpt["in_nc"], use_lpips=True)
    print("Final validation metrics:", metrics)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    examples_dir = os.path.join(args.out_dir, "comparisons")
    os.makedirs(examples_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for lr, gt in DataLoader(val_ds, batch_size=1, shuffle=True):
            if count >= args.n_examples:
                break
            lr, gt = lr.to(device), gt.to(device)
            pred = model(lr).clamp(0, 1)
            make_comparison_grid(lr[0], pred[0], gt[0],
                                  os.path.join(examples_dir, f"compare_{count:02d}.png"))
            count += 1
    print(f"Wrote {count} comparison grids to {examples_dir}")
    print(f"Metrics saved to {os.path.join(args.out_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
