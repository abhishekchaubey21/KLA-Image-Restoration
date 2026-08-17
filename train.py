"""
Training script for the KLA image restoration challenge.

Tuned for a 6GB laptop GPU (RTX 3050):
- small model
- BF16 AMP
- modest batch size
- gradient accumulation available
- EMA validation
- optional extra noise augmentation
- supports normal resume checkpoints
- supports FINE-TUNING directly from best_model.pt

Normal training:
    python train.py --gt_dirs data/GT_0 data/GT_1 --lr_dir data/NoisyLR \
        --epochs 60 --batch_size 8 --model_size small --use_amp

Fine-tuning from best_model.pt:
    python train.py --gt_dirs data/GT_0 data/GT_1 --lr_dir data/NoisyLR \
        --out_dir checkpoints/finetune_B \
        --epochs 30 --batch_size 6 --model_size small \
        --in_nc 1 --scale 2 --lr 2e-5 \
        --amp_dtype bf16 --grad_clip 1.0 \
        --w_ssim 0.4 --use_lpips --w_lpips 0.08 \
        --no_extra_noise_aug \
        --resume checkpoints/best_model.pt
"""

import argparse
import os
import time
import json

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from dataset import RestorationDataset
from model import build_model
from losses import CombinedLoss
from metrics import evaluate


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(
                    v.detach(), alpha=1 - self.decay
                )
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--gt_dirs",
        nargs="+",
        required=True,
        help="e.g. data/GT_0 data/GT_1"
    )

    p.add_argument(
        "--lr_dir",
        required=True,
        help="e.g. data/NoisyLR"
    )

    p.add_argument(
        "--out_dir",
        default="checkpoints"
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=60
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    p.add_argument(
        "--grad_accum",
        type=int,
        default=1
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )

    p.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Upscaling factor"
    )

    p.add_argument(
        "--in_nc",
        type=int,
        default=1,
        help="1=grayscale, 3=RGB"
    )

    p.add_argument(
        "--model_size",
        default="small",
        choices=["tiny", "small", "base"]
    )

    p.add_argument(
        "--w_ssim",
        type=float,
        default=0.2
    )

    p.add_argument(
        "--use_lpips",
        action="store_true",
        help="Use LPIPS perceptual loss"
    )

    p.add_argument(
        "--w_lpips",
        type=float,
        default=0.1
    )

    p.add_argument(
        "--use_amp",
        action="store_true",
        default=True
    )

    p.add_argument(
        "--no_amp",
        dest="use_amp",
        action="store_false"
    )

    p.add_argument(
        "--amp_dtype",
        default="bf16",
        choices=["bf16", "fp16"]
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=4
    )

    p.add_argument(
        "--val_every",
        type=int,
        default=1
    )

    p.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint .pt"
    )

    # Extra synthetic noise augmentation
    p.add_argument(
        "--extra_noise_aug",
        action="store_true",
        default=True
    )

    p.add_argument(
        "--no_extra_noise_aug",
        dest="extra_noise_aug",
        action="store_false",
        help="Disable extra synthetic noise augmentation"
    )

    # EMA
    p.add_argument(
        "--use_ema",
        action="store_true",
        default=True
    )

    p.add_argument(
        "--no_ema",
        dest="use_ema",
        action="store_false"
    )

    p.add_argument(
        "--ema_decay",
        type=float,
        default=0.999
    )

    # Channels-last
    p.add_argument(
        "--channels_last",
        action="store_true",
        default=True
    )

    p.add_argument(
        "--no_channels_last",
        dest="channels_last",
        action="store_false"
    )

    return p.parse_args()


def main():

    # ============================================================
    # ARGUMENTS / DEVICE
    # ============================================================

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    use_cuda = device.type == "cuda"

    if use_cuda:
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        torch.backends.cudnn.benchmark = True

    use_channels_last = (
        args.channels_last and use_cuda
    )

    # ============================================================
    # DATASETS
    # ============================================================

    train_ds = RestorationDataset(
        args.gt_dirs,
        args.lr_dir,
        split="train",
        extra_noise_aug=args.extra_noise_aug
    )

    val_ds = RestorationDataset(
        args.gt_dirs,
        args.lr_dir,
        split="val",
        augment=False,
        extra_noise_aug=False
    )

    print(
        f"[train] {len(train_ds)} samples"
    )

    print(
        f"[val] {len(val_ds)} samples"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ============================================================
    # MODEL
    # ============================================================

    model = build_model(
        in_nc=args.in_nc,
        out_nc=args.in_nc,
        scale=args.scale,
        size=args.model_size
    ).to(device)

    if use_channels_last:
        model = model.to(
            memory_format=torch.channels_last
        )

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Model params: {n_params / 1e6:.2f}M "
        f"({args.model_size})"
    )

    # ============================================================
    # LOSS
    # ============================================================

    criterion = CombinedLoss(
        w_pixel=1.0,
        w_ssim=args.w_ssim,
        w_lpips=(
            args.w_lpips
            if args.use_lpips
            else 0.0
        ),
        in_nc=args.in_nc
    ).to(device)

    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    )

    # ============================================================
    # AMP
    # ============================================================

    amp_dtype = (
        torch.bfloat16
        if args.amp_dtype == "bf16"
        else torch.float16
    )

    use_scaler = (
        args.use_amp
        and args.amp_dtype == "fp16"
    )

    scaler = GradScaler(
        enabled=use_scaler
    )

    # ============================================================
    # EMA
    # ============================================================

    ema = (
        EMA(
            model,
            decay=args.ema_decay
        )
        if args.use_ema
        else None
    )

    ema_eval_model = None

    if args.use_ema:

        ema_eval_model = build_model(
            in_nc=args.in_nc,
            out_nc=args.in_nc,
            scale=args.scale,
            size=args.model_size
        ).to(device)

        if use_channels_last:
            ema_eval_model = ema_eval_model.to(
                memory_format=torch.channels_last
            )

    # ============================================================
    # CHECKPOINT / FINE-TUNING
    # ============================================================

    start_epoch = 0
    best_ssim = -1.0

    if args.resume and os.path.exists(args.resume):

        print()
        print("=" * 60)
        print("LOADING CHECKPOINT")
        print("=" * 60)

        ckpt = torch.load(
            args.resume,
            map_location=device,
            weights_only=True
        )

        # --------------------------------------------------------
        # CASE 1:
        # best_model.pt
        #
        # Contains model weights but normally does NOT contain
        # optimizer state.
        # --------------------------------------------------------

        if "optimizer" not in ckpt:

            print(
                "Checkpoint type: BEST MODEL"
            )

            # Load ONLY model weights.
            model.load_state_dict(
                ckpt["model"]
            )

            # Fresh optimizer.
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-5
            )

            # Fresh scheduler.
            scheduler = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs
                )
            )

            # Start new fine-tuning run.
            start_epoch = 0

            # Reset experiment-specific best.
            best_ssim = -1.0

            # IMPORTANT:
            # Initialize EMA from the loaded model.
            if ema is not None:

                ema.shadow = {
                    k: v.detach().clone()
                    for k, v
                    in model.state_dict().items()
                }

            print(
                f"Fine-tuning from: {args.resume}"
            )

            print(
                f"Fresh optimizer LR: {args.lr}"
            )

        # --------------------------------------------------------
        # CASE 2:
        # last_checkpoint.pt
        #
        # Contains model + optimizer and can be resumed normally.
        # --------------------------------------------------------

        else:

            print(
                "Checkpoint type: FULL TRAINING CHECKPOINT"
            )

            model.load_state_dict(
                ckpt["model"]
            )

            optimizer.load_state_dict(
                ckpt["optimizer"]
            )

            start_epoch = (
                ckpt["epoch"] + 1
            )

            best_ssim = ckpt.get(
                "best_ssim",
                -1.0
            )

            if (
                ema is not None
                and "ema" in ckpt
            ):

                ema.shadow = {
                    k: v.to(device)
                    for k, v
                    in ckpt["ema"].items()
                }

            print(
                f"Resumed from: {args.resume}"
            )

            print(
                f"Starting epoch: {start_epoch}"
            )

    # ============================================================
    # CHECKPOINT SAVER
    # ============================================================

    def save_checkpoint(
        path,
        epoch,
        extra=None
    ):

        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "in_nc": args.in_nc,
            "scale": args.scale,
            "model_size": args.model_size,
            "epoch": epoch,
            "best_ssim": best_ssim
        }

        if ema is not None:
            payload["ema"] = (
                ema.state_dict()
            )

        if extra:
            payload.update(extra)

        torch.save(
            payload,
            path
        )

    # ============================================================
    # TRAINING
    # ============================================================

    history = []

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        t0 = time.time()

        model.train()

        running_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )

        # --------------------------------------------------------
        # TRAINING BATCHES
        # --------------------------------------------------------

        for i, (lr, gt) in enumerate(
            train_loader
        ):

            lr = lr.to(
                device,
                non_blocking=True
            )

            gt = gt.to(
                device,
                non_blocking=True
            )

            if use_channels_last:

                lr = lr.contiguous(
                    memory_format=torch.channels_last
                )

            # ----------------------------------------------------
            # FORWARD
            # ----------------------------------------------------

            with autocast(
                device_type=device.type,
                enabled=args.use_amp,
                dtype=amp_dtype
            ):

                pred = model(lr)

                loss, logs = criterion(
                    pred,
                    gt
                )

                loss = (
                    loss / args.grad_accum
                )

            # ----------------------------------------------------
            # CHECK LOSS
            # ----------------------------------------------------

            if not torch.isfinite(loss):

                print(
                    f"  !! non-finite loss at "
                    f"epoch {epoch} step {i} "
                    f"({loss.item()}) "
                    f"— skipping batch"
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                continue

            # ----------------------------------------------------
            # BACKWARD
            # ----------------------------------------------------

            if use_scaler:

                scaler.scale(
                    loss
                ).backward()

            else:

                loss.backward()

            # ----------------------------------------------------
            # OPTIMIZER STEP
            # ----------------------------------------------------

            if (
                (i + 1)
                % args.grad_accum
                == 0
            ):

                if use_scaler:

                    scaler.unscale_(
                        optimizer
                    )

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip
                    )
                )

                if not torch.isfinite(
                    grad_norm
                ):

                    print(
                        f"  !! non-finite "
                        f"grad norm at "
                        f"epoch {epoch} "
                        f"step {i}"
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    if use_scaler:
                        scaler.update()

                    continue

                if use_scaler:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                # EMA update
                if ema is not None:

                    ema.update(
                        model
                    )

            running_loss += (
                loss.item()
                * args.grad_accum
            )

            # ----------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------

            if (
                device.type == "cuda"
                and i % 50 == 0
            ):

                mem = (
                    torch.cuda.max_memory_allocated()
                    / 1e9
                )

                print(
                    f"  epoch {epoch} "
                    f"step {i}/{len(train_loader)} "
                    f"loss="
                    f"{loss.item() * args.grad_accum:.4f} "
                    f"peak_vram="
                    f"{mem:.2f}GB"
                )

        # ========================================================
        # END OF EPOCH
        # ========================================================

        scheduler.step()

        avg_loss = (
            running_loss
            / len(train_loader)
        )

        dt = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "time_s": dt,
            "lr": scheduler.get_last_lr()[0]
        }

        # ========================================================
        # VALIDATION
        # ========================================================

        if (
            (epoch + 1) % args.val_every == 0
            or epoch == args.epochs - 1
        ):

            # ----------------------------------------------------
            # RAW MODEL
            # ----------------------------------------------------

            val_metrics = evaluate(
                model,
                val_loader,
                device,
                in_nc=args.in_nc,
                use_lpips=args.use_lpips
            )

            log_entry.update(
                val_metrics
            )

            print(
                f"[Epoch {epoch}] "
                f"loss={avg_loss:.4f} "
                f"time={dt:.1f}s | "
                f"val PSNR="
                f"{val_metrics['psnr']:.2f} "
                f"SSIM="
                f"{val_metrics['ssim']:.4f} "
                + (
                    f"LPIPS="
                    f"{val_metrics.get('lpips', 0):.4f}"
                    if args.use_lpips
                    else ""
                )
            )

            # ----------------------------------------------------
            # RAW IS INITIAL CANDIDATE
            # ----------------------------------------------------

            candidate_ssim = (
                val_metrics["ssim"]
            )

            candidate_state = (
                model.state_dict()
            )

            tag = "raw"

            # ----------------------------------------------------
            # EMA VALIDATION
            # ----------------------------------------------------

            if ema is not None:

                ema_eval_model.load_state_dict(
                    ema.state_dict()
                )

                ema_metrics = evaluate(
                    ema_eval_model,
                    val_loader,
                    device,
                    in_nc=args.in_nc,
                    use_lpips=args.use_lpips
                )

                log_entry.update({
                    f"ema_{k}": v
                    for k, v
                    in ema_metrics.items()
                })

                print(
                    f"           [EMA] "
                    f"PSNR="
                    f"{ema_metrics['psnr']:.2f} "
                    f"SSIM="
                    f"{ema_metrics['ssim']:.4f} "
                    + (
                        f"LPIPS="
                        f"{ema_metrics.get('lpips', 0):.4f}"
                        if args.use_lpips
                        else ""
                    )
                )

                if (
                    ema_metrics["ssim"]
                    > candidate_ssim
                ):

                    candidate_ssim = (
                        ema_metrics["ssim"]
                    )

                    candidate_state = (
                        ema.state_dict()
                    )

                    tag = "ema"

            # ----------------------------------------------------
            # SAVE BEST MODEL
            # ----------------------------------------------------

            if candidate_ssim > best_ssim:

                best_ssim = candidate_ssim

                torch.save(
                    {
                        "model": candidate_state,
                        "in_nc": args.in_nc,
                        "scale": args.scale,
                        "model_size": args.model_size,
                        "epoch": epoch,
                        "best_ssim": best_ssim,
                        "source": tag
                    },
                    os.path.join(
                        args.out_dir,
                        "best_model.pt"
                    )
                )

                print(
                    f"  -> new best "
                    f"(SSIM={best_ssim:.4f}, "
                    f"source={tag}), "
                    f"saved best_model.pt"
                )

        else:

            print(
                f"[Epoch {epoch}] "
                f"loss={avg_loss:.4f} "
                f"time={dt:.1f}s"
            )

        # ========================================================
        # SAVE FULL CHECKPOINT
        # ========================================================

        history.append(
            log_entry
        )

        save_checkpoint(
            os.path.join(
                args.out_dir,
                "last_checkpoint.pt"
            ),
            epoch
        )

        # ========================================================
        # HISTORY
        # ========================================================

        with open(
            os.path.join(
                args.out_dir,
                "history.json"
            ),
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )

    # ============================================================
    # FINISHED
    # ============================================================

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best val SSIM: {best_ssim:.4f}"
    )

    print(
        "Best weights:",
        os.path.join(
            args.out_dir,
            "best_model.pt"
        )
    )


if __name__ == "__main__":
    main()