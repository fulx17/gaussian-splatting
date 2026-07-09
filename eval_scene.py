import os
import json
from pathlib import Path
from argparse import ArgumentParser
import lpips

import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


def read_image(path):
    img = Image.open(path)
    return tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()


def eval_scene(gt_path, sample_path):
    gt_path = Path(gt_path)
    sample_path = Path(sample_path)

    if not gt_path.exists():
        raise FileNotFoundError(f"gt_path không tồn tại: {gt_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"sample_path không tồn tại: {sample_path}")

    gt_files = sorted(os.listdir(gt_path))
    sample_files = sorted(os.listdir(sample_path))

    if len(gt_files) != len(sample_files):
        print(f"!!! Warning: số lượng ảnh lệch nhau: gt={len(gt_files)}, sample={len(sample_files)}")

    common = sorted(set(gt_files) & set(sample_files))
    if len(common) == 0:
        raise RuntimeError(f"Không có file khớp tên giữa {gt_path} và {sample_path}")

    ssims, psnrs, lpipss = [], [], []

    for fname in tqdm(common, desc="Evaluating"):
        gt_img = read_image(gt_path / fname)
        sample_img = read_image(sample_path / fname)

        ssims.append(ssim(sample_img, gt_img).item())
        psnrs.append(psnr(sample_img, gt_img).item())
        lpipss.append(lpips(sample_img, gt_img, net_type='alex').item())

    return {
        "num_images": len(common),
        "SSIM": sum(ssims) / len(ssims),
        "PSNR": sum(psnrs) / len(psnrs),
        "LPIPS": sum(lpipss) / len(lpipss),
    }


def compute_weighted_score(ssim_val, psnr_val, lpips_val, psnr_max=40.0):
    psnr_norm = torch.clamp(torch.tensor(psnr_val) / psnr_max, 0.0, 1.0).item()
    score = 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm
    return score, psnr_norm


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a single scene")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--image_dir", default="/kaggle/working/image_outputs")
    parser.add_argument("--eval_dir", default="/kaggle/working/eval_outputs")
    parser.add_argument("--psnr_max", type=float, default=40.0)
    args = parser.parse_args()

    gt_path = os.path.join(args.input_dir, args.scene_name, "test", "images")
    sample_path = os.path.join(args.image_dir, args.scene_name)

    metrics = eval_scene(gt_path, sample_path)

    weighted_score, psnr_norm = compute_weighted_score(
        metrics["SSIM"], metrics["PSNR"], metrics["LPIPS"], args.psnr_max
    )

    result = {
        "scene_name": args.scene_name,
        "gt_path": gt_path,
        "sample_path": sample_path,
        **metrics,
        "psnr_norm": psnr_norm,
        "psnr_max": args.psnr_max,
        "weighted_score": weighted_score,
    }

    print(json.dumps(result, indent=2))

    os.makedirs(args.eval_dir, exist_ok=True)
    output_json = os.path.join(args.eval_dir, f"{args.scene_name}.json")
    with open(output_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved to {output_json}")