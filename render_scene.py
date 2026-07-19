import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene.cameras import Camera
from scene.colmap_loader import qvec2rotmat, read_intrinsics_binary
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration
from utils.general_utils import safe_state
import torch.nn.functional as F

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def camera_from_csv_row(row, idx, data_device, width, height, fx, fy):
    qvec = np.array(
        [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
        dtype=np.float64,
    )
    tvec = np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])], dtype=np.float64)
    rotation_world_to_camera = qvec2rotmat(qvec)

    dummy = Image.new("RGB", (width, height), (0, 0, 0))
    return Camera(
        resolution=(width, height),
        colmap_id=idx,
        R=rotation_world_to_camera.T,
        T=tvec,
        FoVx=focal2fov(fx, width),
        FoVy=focal2fov(fy, height),
        depth_params=None,
        image=dummy,
        invdepthmap=None,
        image_name=Path(row["image_name"]).name,
        uid=idx,
        data_device=data_device,
    )


def load_gaussians(dataset, iteration):
    gaussians = GaussianModel(dataset.sh_degree)
    loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud")) if iteration == -1 else iteration
    ply_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{loaded_iter}",
        "point_cloud.ply",
    )
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Cannot find trained point cloud: {ply_path}")
    print(f"Loading trained model at iteration {loaded_iter}")
    gaussians.load_ply(ply_path, dataset.train_test_exp)
    return gaussians, loaded_iter


def load_distortion_params(orig_dir, scene_name):
    cameras_bin = Path(orig_dir) / scene_name / "train" / "sparse" / "0" / "cameras.bin"
    cams = read_intrinsics_binary(str(cameras_bin))
    assert len(cams) == 1, f"Expect exactly 1 camera, got {len(cams)}"
    cam = next(iter(cams.values()))
    assert cam.model == "SIMPLE_RADIAL", f"Unsupported model: {cam.model}"
    f, cx, cy, k = cam.params
    return dict(f=float(f), cx=float(cx), cy=float(cy), k=float(k),
                width=cam.width, height=cam.height)


def load_undistorted_camera_params(input_dir, scene_name):
    cameras_bin = Path(input_dir) / scene_name / "train" / "sparse" / "0" / "cameras.bin"
    cams = read_intrinsics_binary(str(cameras_bin))
    assert len(cams) == 1, f"Expect exactly 1 camera, got {len(cams)}"
    cam = next(iter(cams.values()))
    assert cam.model == "PINHOLE", (
        f"Camera trong {cameras_bin} phai la PINHOLE (da undistort), gap {cam.model}."
    )
    fx, fy, cx, cy = cam.params
    assert abs(fx - fy) < 1e-3, f"fx != fy sau undistort ({fx} vs {fy})"
    return dict(f=float(fx), cx=float(cx), cy=float(cy),
                width=cam.width, height=cam.height)


# ===== THAY ĐỔI 1: redistort_image hỗ trợ chọn interpolation (bilinear/bicubic) =====
def redistort_image(img, f, cx, cy, k, num_iters=15, interpolation="bicubic"):
    """img: tensor [C,H,W] -> anh da meo theo k, dung interpolation duoc chon."""
    C, H, W = img.shape
    device = img.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xd = (xs - cx) / f
    yd = (ys - cy) / f
    rd = torch.sqrt(xd * xd + yd * yd)

    ru = rd.clone()
    for _ in range(num_iters):
        g = k * ru**3 + ru - rd
        g_prime = 3 * k * ru**2 + 1
        g_prime = torch.where(g_prime.abs() < 1e-12, torch.full_like(g_prime, 1e-12), g_prime)
        ru = ru - g / g_prime

    scale = torch.where(rd > 1e-12, ru / rd, torch.ones_like(rd))
    xu = xd * scale
    yu = yd * scale

    u_src = xu * f + cx
    v_src = yu * f + cy

    grid_x = (u_src / (W - 1)) * 2 - 1
    grid_y = (v_src / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    out = F.grid_sample(
        img.unsqueeze(0), grid, mode=interpolation,
        padding_mode="zeros", align_corners=True,
    )
    # Bicubic co the vuot nguon 0-1, can clamp lai
    return out.squeeze(0).clamp(0.0, 1.0)


def redistort_and_crop(img, f, cx_render, cy_render, k, cx_orig, cy_orig, orig_w, orig_h,
                        num_iters=15, interpolation="bicubic"):
    distorted_full = redistort_image(
        img, f=f, cx=cx_render, cy=cy_render, k=k,
        num_iters=num_iters, interpolation=interpolation,
    )

    _, H, W = distorted_full.shape
    offset_x = int(round(cx_render - cx_orig))
    offset_y = int(round(cy_render - cy_orig))

    x0 = max(offset_x, 0)
    y0 = max(offset_y, 0)
    x1 = min(offset_x + orig_w, W)
    y1 = min(offset_y + orig_h, H)

    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"Crop window rong/vuot canvas: offset=({offset_x},{offset_y}) "
            f"canvas=({W}x{H}) target=({orig_w}x{orig_h})"
        )

    cropped = distorted_full[:, y0:y1, x0:x1]

    if cropped.shape[1] != orig_h or cropped.shape[2] != orig_w:
        pad_top = y0 - offset_y
        pad_left = x0 - offset_x
        pad_bottom = orig_h - cropped.shape[1] - pad_top
        pad_right = orig_w - cropped.shape[2] - pad_left
        cropped = F.pad(
            cropped, (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant", value=0,
        )

    return cropped


# ===== THAY ĐỔI 2: thêm unsharp mask (sharpen) trên tensor float32 =====
def unsharp_mask(image, amount=0.0, sigma=0.7):
    """Ap dung Gaussian unsharp mask tren tensor CHW, float32, gia tri 0-1."""
    if float(amount) == 0.0:
        return image

    channels, height, width = image.shape
    radius = max(int(np.ceil(3.0 * float(sigma))), 1)
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    weights = kernel_2d.view(1, 1, *kernel_2d.shape).expand(channels, 1, -1, -1).contiguous()

    padding_mode = "reflect" if height > radius and width > radius else "replicate"
    padded = F.pad(image.unsqueeze(0), (radius, radius, radius, radius), mode=padding_mode)
    blurred = F.conv2d(padded, weights, groups=channels).squeeze(0)
    return torch.clamp(image + float(amount) * (image - blurred), 0.0, 1.0)


# ===== THAY ĐỔI 3: lưu ảnh với cấu hình JPEG tối ưu (thay cho torchvision.save_image) =====
def save_render(rendering_tensor, out_path, output_format, jpeg_quality=96,
                 jpeg_subsampling=0, jpeg_optimize=True):
    """
    rendering_tensor: tensor [C,H,W], float32, gia tri 0-1
    Ho tro them: 'png16' (PNG 16-bit) va 'npy32' (float32 raw, khong lossy)
    """
    suffix = Path(out_path).suffix.lower()

    # ===== NPY 32-bit: luu float32 raw, khong quantize =====
    if suffix == ".npy":
        arr = rendering_tensor.detach().cpu().clamp(0, 1).numpy().astype(np.float32)
        # luu HWC cho de doc lai bang cv2/np thong nhat voi cac dinh dang khac
        arr = np.transpose(arr, (1, 2, 0))
        np.save(out_path, arr)
        return

    # ===== PNG 16-bit =====
    if suffix == ".png16" or (suffix == ".png" and output_format == "png16"):
        img_np = rendering_tensor.detach().cpu().clamp(0, 1).mul(65535).round()
        img_np = img_np.numpy().astype(np.uint16)
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
        # PIL khong ho tro luu PNG 16-bit RGB truc tiep tot bang cv2, dung cv2 thay the
        import cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        return

    # ===== PNG 8-bit (giu nguyen nhu cu) =====
    img_np = rendering_tensor.detach().cpu().clamp(0, 1).mul(255).round().byte()
    img_np = img_np.permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(img_np, mode="RGB")

    if suffix in (".jpg", ".jpeg"):
        pil_img.save(
            out_path, format="JPEG",
            quality=jpeg_quality, subsampling=jpeg_subsampling, optimize=jpeg_optimize,
        )
    else:
        pil_img.save(out_path, format="PNG")


def render_scene(dataset, pipeline, input_dir, output_dir, scene_name, iteration, orig_dir,
                  output_format="original", redistort_interpolation="bicubic",
                  sharpen_amount=0.3, sharpen_sigma=0.7,
                  jpeg_quality=96, jpeg_subsampling=0, jpeg_optimize=True):
    gaussians, loaded_iter = load_gaussians(dataset, iteration)
    scene_dir = Path(output_dir) / scene_name
    test_poses_csv = Path(input_dir) / scene_name / "test" / "test_poses.csv"
    scene_dir.mkdir(parents=True, exist_ok=True)

    dist = load_distortion_params(orig_dir, scene_name)
    print(f"[{scene_name}] distortion k={dist['k']:.6f} f={dist['f']:.2f} "
          f"cx={dist['cx']:.2f} cy={dist['cy']:.2f} size=({dist['width']}x{dist['height']})")

    und = load_undistorted_camera_params(input_dir, scene_name)
    print(f"[{scene_name}] undistorted render canvas f={und['f']:.2f} "
          f"cx={und['cx']:.2f} cy={und['cy']:.2f} size=({und['width']}x{und['height']})")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with open(test_poses_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    with torch.no_grad():
        for idx, row in enumerate(tqdm(rows, desc=f"Rendering {scene_name}")):
            camera = camera_from_csv_row(
                row, idx, dataset.data_device,
                width=und["width"], height=und["height"],
                fx=und["f"], fy=und["f"],
            )
            rendering = render(
                camera,
                gaussians,
                pipeline,
                background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"]

            # ===== redistort dung interpolation da chon (mac dinh bicubic) =====
            if abs(dist["k"]) > 1e-8:
                rendering = redistort_and_crop(
                    rendering,
                    f=und["f"],
                    cx_render=und["cx"],
                    cy_render=und["cy"],
                    k=dist["k"],
                    cx_orig=dist["cx"],
                    cy_orig=dist["cy"],
                    orig_w=dist["width"],
                    orig_h=dist["height"],
                    interpolation=redistort_interpolation,
                )

            # ===== sharpen truoc khi luu, van tren tensor float32 =====
            rendering = unsharp_mask(rendering, amount=sharpen_amount, sigma=sharpen_sigma)

            if output_format == "png":
                out_name = Path(row["image_name"]).stem + ".png"
            elif output_format == "png16":
                out_name = Path(row["image_name"]).stem + ".png"   # vẫn đuôi .png, phân biệt bằng output_format truyền vào
            elif output_format == "npy32":
                out_name = Path(row["image_name"]).stem + ".npy"
            elif output_format == "jpeg":
                out_name = Path(row["image_name"]).stem + ".jpg"
            else:
                out_name = row["image_name"]

            out_path = scene_dir / out_name
            save_render(
                rendering, out_path, output_format,
                jpeg_quality=jpeg_quality,
                jpeg_subsampling=jpeg_subsampling,
                jpeg_optimize=jpeg_optimize,
            )
            del camera, rendering

    print(f"Rendered {len(rows)} images for {scene_name} from iteration {loaded_iter} -> {scene_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render VAR scene with trained 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--orig_dir", default="/kaggle/input/datasets/xuanph/phase1/phase1/private_set1")
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--image_dir", default="/kaggle/working/image_outputs")
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--output_format",
        default="jpeg",
        choices=["original", "png", "png16", "npy32", "jpeg"],
        help="'original' giu duoi file goc, 'png' ep ve .png 8-bit, "
            "'png16' ep ve .png 16-bit, 'npy32' luu float32 raw .npy, "
            "'jpeg' ep ve .jpg voi cau hinh toi uu"
    )
    parser.add_argument("--redistort_interpolation", choices=["bilinear", "bicubic"], default="bicubic")
    parser.add_argument("--sharpen_amount", type=float, default=0.3)
    parser.add_argument("--sharpen_sigma", type=float, default=0.7)
    parser.add_argument("--jpeg_quality", type=int, default=96)
    parser.add_argument("--jpeg_subsampling", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--jpeg_optimize", action="store_true", default=True)

    args = get_combined_args(parser)

    safe_state(args.quiet)

    os.makedirs(args.image_dir, exist_ok=True)

    render_scene(
        model.extract(args),
        pipeline.extract(args),
        args.input_dir,
        args.image_dir,
        args.scene_name,
        args.iterations,
        args.orig_dir,
        output_format=args.output_format,
        redistort_interpolation=args.redistort_interpolation,
        sharpen_amount=args.sharpen_amount,
        sharpen_sigma=args.sharpen_sigma,
        jpeg_quality=args.jpeg_quality,
        jpeg_subsampling=args.jpeg_subsampling,
        jpeg_optimize=args.jpeg_optimize,
    )