#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from PIL import Image


def save_render_jpeg(
    image,
    output_path,
    quality=95,
    subsampling=2,
    optimize=False,
):
    """Save a CHW float render with explicit JPEG encoding controls.

    The defaults preserve the historical Q95 4:2:0 output.  Passing
    ``subsampling=0`` keeps full-resolution chroma (4:4:4), while Pillow's
    ``optimize`` option reduces file size through a better Huffman table
    without changing the quantized image coefficients.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB CHW tensor, got shape {tuple(image.shape)}")
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")
    if subsampling not in (0, 1, 2):
        raise ValueError(
            "JPEG subsampling must be 0 (4:4:4), 1 (4:2:2), or 2 (4:2:0), "
            f"got {subsampling}"
        )
    if not isinstance(optimize, bool):
        raise TypeError(f"JPEG optimize must be bool, got {type(optimize).__name__}")

    image_u8 = (
        image.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .add(0.5)
        .to(device="cpu", dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    Image.fromarray(image_u8).save(
        output_path,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=optimize,
    )


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
