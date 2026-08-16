"""
STANDALONE INFERENCE / EVALUATION SCRIPT — this is the file KLA runs
AS-IS on their H100 to benchmark your submission.

Usage:
    python evaluate_submission.py --input_dir /path/to/test_images \
        --output_dir /path/to/restored_outputs \
        --checkpoint checkpoints/best_model.pt
"""

import argparse
import os
import time
import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import torch

from model import build_model
from dataset import load_array, IMG_EXTS


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_dir",
        required=True,
        help="Directory of degraded test images"
    )

    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write restored images"
    )

    p.add_argument(
        "--checkpoint",
        default="checkpoints/best_model.pt"
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for H100 inference"
    )

    p.add_argument(
        "--device",
        default=None,
        help="cuda / cpu; auto-detects if unset"
    )

    p.add_argument(
        "--num_io_workers",
        type=int,
        default=8,
        help="Threads for loading and saving files"
    )

    p.add_argument(
        "--png_compress_level",
        type=int,
        default=1,
        help="PNG compression level: 0 fastest, 9 smallest"
    )

    return p.parse_args()


def list_input_files(input_dir):
    """
    Find all supported input files.

    Supports normal image formats as well as .npy.
    """

    files = []

    for ext in IMG_EXTS | {".npy"}:
        files += glob.glob(
            os.path.join(input_dir, f"*{ext}")
        )

    return sorted(files)


def _load_one(path):
    """
    Load one image / numpy array and convert it to
    a CHW float tensor.
    """

    a = load_array(path)

    # H,W -> H,W,1
    if a.ndim == 2:
        a = a[:, :, None]

    # H,W,C -> C,H,W
    tensor = torch.from_numpy(
        np.ascontiguousarray(
            np.transpose(a, (2, 0, 1))
        )
    ).float()

    return path, tensor


def save_output(arr, path, compress_level=1):
    """
    Convert CHW float32 array in [0,1] to PNG.
    """

    # CHW -> HWC
    arr = np.transpose(arr, (1, 2, 0))

    # Grayscale
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]

    # [0,1] -> [0,255]
    img = (
        arr * 255.0
    ).round().astype(np.uint8)

    Image.fromarray(img).save(
        path,
        compress_level=compress_level
    )


def main():

    args = parse_args()

    # ---------------------------------------------------------
    # OUTPUT DIRECTORY
    # ---------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # DEVICE
    # ---------------------------------------------------------

    device = (
        torch.device(args.device)
        if args.device
        else torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    print("Device:", device)

    use_cuda = device.type == "cuda"

    if use_cuda:

        torch.backends.cudnn.benchmark = True

        torch.backends.cuda.matmul.allow_tf32 = True

        torch.backends.cudnn.allow_tf32 = True

    # ---------------------------------------------------------
    # START TIMER
    # ---------------------------------------------------------

    t_start = time.time()

    # ---------------------------------------------------------
    # LOAD CHECKPOINT
    # ---------------------------------------------------------

    print(
        f"Loading checkpoint: {args.checkpoint}"
    )

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )

    # ---------------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------------

    model = build_model(
        in_nc=ckpt["in_nc"],
        out_nc=ckpt["in_nc"],
        scale=ckpt["scale"],
        size=ckpt["model_size"]
    )

    model.load_state_dict(
        ckpt["model"]
    )

    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # CHANNELS LAST
    # ---------------------------------------------------------

    if use_cuda:
        model = model.to(
            memory_format=torch.channels_last
        )

    t_model_ready = time.time()

    # ---------------------------------------------------------
    # FIND INPUT FILES
    # ---------------------------------------------------------

    files = list_input_files(
        args.input_dir
    )

    if not files:

        raise RuntimeError(
            f"No input images found in {args.input_dir}"
        )

    print(
        f"Found {len(files)} input files"
    )

    # ---------------------------------------------------------
    # THREAD POOL
    # ---------------------------------------------------------

    io_pool = ThreadPoolExecutor(
        max_workers=args.num_io_workers
    )

    # ---------------------------------------------------------
    # LOAD INPUTS
    # ---------------------------------------------------------

    print("Loading input files...")

    loaded = list(
        io_pool.map(
            _load_one,
            files
        )
    )

    # ---------------------------------------------------------
    # GROUP BY IMAGE SIZE
    # ---------------------------------------------------------

    groups = defaultdict(list)

    for path, tensor in loaded:

        groups[
            tuple(tensor.shape)
        ].append(
            (path, tensor)
        )

    print(
        f"Found {len(groups)} shape group(s)"
    )

    for shape, items in groups.items():

        print(
            f"Shape {shape}: {len(items)} images"
        )

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------

    write_futures = []

    print("Running inference...")

    with torch.inference_mode():

        for shape, items in groups.items():

            for i in range(
                0,
                len(items),
                args.batch_size
            ):

                chunk = items[
                    i:i + args.batch_size
                ]

                paths = [
                    p
                    for p, _ in chunk
                ]

                batch = torch.stack(
                    [
                        t
                        for _, t in chunk
                    ]
                ).to(
                    device,
                    non_blocking=True
                )

                # -------------------------------------------------
                # CHANNELS LAST INPUT
                # -------------------------------------------------

                if use_cuda:

                    batch = batch.contiguous(
                        memory_format=torch.channels_last
                    )

                # -------------------------------------------------
                # MIXED PRECISION
                # -------------------------------------------------

                with torch.autocast(
                    device_type=device.type,
                    enabled=use_cuda,
                    dtype=torch.bfloat16
                ):

                    preds = model(batch)

                # -------------------------------------------------
                # POST PROCESS
                # -------------------------------------------------

                preds = (
                    preds
                    .float()
                    .clamp(0, 1)
                    .cpu()
                    .numpy()
                )

                # -------------------------------------------------
                # SAVE OUTPUTS
                # -------------------------------------------------

                for pred, src_path in zip(
                    preds,
                    paths
                ):

                    base = os.path.splitext(
                        os.path.basename(src_path)
                    )[0]

                    out_path = os.path.join(
                        args.output_dir,
                        base + ".png"
                    )

                    write_futures.append(
                        io_pool.submit(
                            save_output,
                            pred,
                            out_path,
                            args.png_compress_level
                        )
                    )

    # ---------------------------------------------------------
    # WAIT FOR FILE WRITES
    # ---------------------------------------------------------

    for future in write_futures:

        future.result()

    io_pool.shutdown()

    # ---------------------------------------------------------
    # TIMING
    # ---------------------------------------------------------

    t_end = time.time()

    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)

    print(
        f"Processed {len(files)} images"
    )

    print(
        f"Shape groups: {len(groups)}"
    )

    print(
        f"Model load time: "
        f"{t_model_ready - t_start:.3f}s"
    )

    print(
        f"Inference + I/O time: "
        f"{t_end - t_model_ready:.3f}s"
    )

    print(
        f"Total end-to-end time: "
        f"{t_end - t_start:.3f}s"
    )

    print(
        f"Outputs saved to: "
        f"{args.output_dir}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()