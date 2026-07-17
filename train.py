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

import os
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from lpipsPyTorch import lpips

import csv
import random
import torchvision
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor

from render_scene import (
    camera_from_csv_row, load_distortion_params,
    load_undistorted_camera_params, redistort_and_crop,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# ============================================================
# VAR: shared helpers cho HF evaluation (loss va gradient)
# ============================================================
def load_gt_image(gt_dir, image_name, device):
    stem = Path(image_name).stem
    gt_dir = Path(gt_dir)
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        gt_path = gt_dir / f"{stem}{ext}"
        if gt_path.exists():
            img = Image.open(gt_path).convert("RGB")
            return to_tensor(img).to(device)
    return None


def load_hf_map(hf_masks_dir, image_name):
    stem = Path(image_name).stem
    hf_path = Path(hf_masks_dir) / f"{stem}_highfreq.png"
    if not hf_path.exists():
        return None
    return np.array(Image.open(hf_path).convert("L"), dtype=np.float32)


def hf_quintile_bounds(hf_map):
    """Tra ve bounds [-1, q20, q40, q60, q80, 256] theo tung anh."""
    qs = np.percentile(hf_map, [20, 40, 60, 80])
    return [-1.0] + list(qs) + [256.0]


def aggregate_quintile(value_map_2d, hf_map):
    """
    value_map_2d: HxW numpy, gia tri khong am (vd |E| hoac |grad|)
    hf_map: HxW numpy [0,255]
    Return: (means[5], contribs[5])
    """
    bounds = hf_quintile_bounds(hf_map)
    total = value_map_2d.sum() + 1e-8
    means, contribs = [], []
    for k in range(5):
        mask = (hf_map > bounds[k]) & (hf_map <= bounds[k + 1])
        if mask.sum() == 0:
            means.append(0.0); contribs.append(0.0); continue
        means.append(float(value_map_2d[mask].mean()))
        contribs.append(float(value_map_2d[mask].sum() / total))
    return means, contribs


def _resolve_scene_paths(dataset, orig_dir, hf_masks_dir):
    source_path = Path(dataset.source_path)
    scene_name = source_path.parent.name
    input_dir = source_path.parent.parent
    scene_dir = input_dir / scene_name

    test_poses_csv = scene_dir / "test" / "test_poses.csv"
    gt_dir = scene_dir / "test" / "images"
    if hf_masks_dir is None:
        hf_masks_dir = scene_dir / "test" / "hf_masks"

    dist = load_distortion_params(orig_dir, scene_name)
    und = load_undistorted_camera_params(input_dir, scene_name)
    return test_poses_csv, gt_dir, hf_masks_dir, dist, und


def _write_csv_row(csv_path, header, iteration, avg_means, avg_contribs):
    write_header = not os.path.exists(csv_path)
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        if write_header:
            f.write(header + "\n")
        row_vals = [str(iteration)] + [f"{v:.6f}" for v in avg_means] + [f"{v:.6f}" for v in avg_contribs]
        f.write(",".join(row_vals) + "\n")


# ============================================================
# VAR: HF quintile LOSS evaluation (khong can gradient)
# ============================================================
def compute_hf_metrics(gt, render_img, hf_map):
    """gt, render_img: CxHxW tensor [0,1]; hf_map: HxW numpy [0,255]. Return (losses[5], contribs[5])."""
    E = (gt - render_img).abs().mean(dim=0).detach().cpu().numpy()  # HxW L1
    return aggregate_quintile(E, hf_map)


def evaluate_hf(dataset, gaussians, pipe, background, iteration,
                 orig_dir, hf_masks_dir, hf_log_csv,
                 num_samples=20, seed=42):
    """Render sample cố định tu test_poses.csv, redistort ve dung size GT,
    tinh HF quintile L1 loss/contribution, ghi 1 dong vao hf_log_csv."""
    test_poses_csv, gt_dir, hf_masks_dir, dist, und = _resolve_scene_paths(dataset, orig_dir, hf_masks_dir)

    if not test_poses_csv.exists():
        print(f"[HF EVAL] Khong tim thay {test_poses_csv}, bo qua.", flush=True)
        return

    with open(test_poses_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    rng = random.Random(seed)
    sample_rows = rng.sample(rows, min(num_samples, len(rows)))

    device = dataset.data_device
    losses_sum = [0.0] * 5
    contribs_sum = [0.0] * 5
    n_ok = 0

    with torch.no_grad():
        for idx, row in enumerate(sample_rows):
            camera = camera_from_csv_row(
                row, idx, device,
                width=und["width"], height=und["height"],
                fx=und["f"], fy=und["f"],
            )
            rendering = render(
                camera, gaussians, pipe, background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"]

            if abs(dist["k"]) > 1e-8:
                rendering = redistort_and_crop(
                    rendering,
                    f=und["f"], cx_render=und["cx"], cy_render=und["cy"],
                    k=dist["k"], cx_orig=dist["cx"], cy_orig=dist["cy"],
                    orig_w=dist["width"], orig_h=dist["height"],
                )

            gt_image = load_gt_image(gt_dir, row["image_name"], device)
            hf_map = load_hf_map(hf_masks_dir, row["image_name"])

            if gt_image is None or hf_map is None:
                del camera, rendering
                continue

            image_c = torch.clamp(rendering, 0.0, 1.0)
            gt_image_c = torch.clamp(gt_image, 0.0, 1.0)

            if image_c.shape != gt_image_c.shape or hf_map.shape != image_c.shape[-2:]:
                print(f"[HF EVAL] Bo qua {row['image_name']}: shape khong khop.", flush=True)
                del camera, rendering
                continue

            l, c = compute_hf_metrics(gt_image_c, image_c, hf_map)
            for k in range(5):
                losses_sum[k] += l[k]
                contribs_sum[k] += c[k]
            n_ok += 1

            del camera, rendering

    if n_ok == 0:
        print(f"[HF EVAL] Khong co anh nao khop GT/HF mask (gt_dir={gt_dir}, hf_masks_dir={hf_masks_dir})", flush=True)
        return

    avg_losses = [v / n_ok for v in losses_sum]
    avg_contribs = [v / n_ok for v in contribs_sum]
    header = ("iter,Q1_loss,Q2_loss,Q3_loss,Q4_loss,Q5_loss,"
              "Q1_contrib,Q2_contrib,Q3_contrib,Q4_contrib,Q5_contrib")
    _write_csv_row(hf_log_csv, header, iteration, avg_losses, avg_contribs)

    print(f"[HF EVAL] iter={iteration} n={n_ok} Q1_loss={avg_losses[0]:.4f} Q5_loss={avg_losses[4]:.4f}", flush=True)


# ============================================================
# VAR: HF quintile GRADIENT evaluation (CAN gradient)
# ============================================================
def evaluate_hf_gradient(dataset, gaussians, pipe, background, iteration,
                          orig_dir, hf_masks_dir, hf_grad_log_csv,
                          num_samples=20, seed=42):
    """
    Render voi gradient bat, backward L1 loss, lay |grad| cua anh render,
    chia theo HF quintile (giong evaluate_hf nhung do GRADIENT thay vi LOSS).

    QUAN TRONG: ham nay se sinh gradient tren tham so Gaussian (vi rendering
    duoc tao ra tu gaussians). Sau khi xong PHAI zero_grad() de khong lam
    hong gradient cua iteration training tiep theo. Goi ham nay sau khi
    optimizer.step() + zero_grad() cua iteration training da hoan tat.
    """
    test_poses_csv, gt_dir, hf_masks_dir, dist, und = _resolve_scene_paths(dataset, orig_dir, hf_masks_dir)

    if not test_poses_csv.exists():
        print(f"[HF GRAD] Khong tim thay {test_poses_csv}, bo qua.", flush=True)
        return

    with open(test_poses_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    rng = random.Random(seed)
    sample_rows = rng.sample(rows, min(num_samples, len(rows)))

    device = dataset.data_device
    grads_sum = [0.0] * 5
    contribs_sum = [0.0] * 5
    n_ok = 0

    was_training = gaussians._xyz.requires_grad  # tham khao, khong bat buoc dung

    with torch.enable_grad():
        for idx, row in enumerate(sample_rows):
            camera = camera_from_csv_row(
                row, idx, device,
                width=und["width"], height=und["height"],
                fx=und["f"], fy=und["f"],
            )
            render_pkg = render(
                camera, gaussians, pipe, background,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            rendering = render_pkg["render"]

            if abs(dist["k"]) > 1e-8:
                rendering = redistort_and_crop(
                    rendering,
                    f=und["f"], cx_render=und["cx"], cy_render=und["cy"],
                    k=dist["k"], cx_orig=dist["cx"], cy_orig=dist["cy"],
                    orig_w=dist["width"], orig_h=dist["height"],
                )

            gt_image = load_gt_image(gt_dir, row["image_name"], device)
            hf_map = load_hf_map(hf_masks_dir, row["image_name"])

            if gt_image is None or hf_map is None:
                continue

            image_c = torch.clamp(rendering, 0.0, 1.0)
            gt_image_c = torch.clamp(gt_image, 0.0, 1.0)

            if image_c.shape != gt_image_c.shape or hf_map.shape != image_c.shape[-2:]:
                print(f"[HF GRAD] Bo qua {row['image_name']}: shape khong khop.", flush=True)
                continue

            # can gradient tren chinh tensor render (khong phai anh da clamp/detach)
            rendering.retain_grad()

            eval_loss = l1_loss(image_c, gt_image_c)  # E = |Render - GT|, dung loss dang dung
            eval_loss.backward()

            if rendering.grad is None:
                print(f"[HF GRAD] Khong co grad cho {row['image_name']}, bo qua.", flush=True)
                # van phai don grad neu co roi rac tren gaussians
                gaussians.optimizer.zero_grad(set_to_none=True)
                continue

            G = rendering.grad.detach().abs().mean(dim=0).cpu().numpy()  # HxW

            g_means, g_contribs = aggregate_quintile(G, hf_map)
            for k in range(5):
                grads_sum[k] += g_means[k]
                contribs_sum[k] += g_contribs[k]
            n_ok += 1

            # QUAN TRONG: don grad tich luy tren tham so Gaussian ngay sau moi sample,
            # de khong anh huong iteration training tiep theo.
            gaussians.optimizer.zero_grad(set_to_none=True)
            if hasattr(gaussians, "exposure_optimizer"):
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)

    # don grad lan cuoi cho chac (phong truong hop loop rong hoac loi giua chung)
    gaussians.optimizer.zero_grad(set_to_none=True)
    if hasattr(gaussians, "exposure_optimizer"):
        gaussians.exposure_optimizer.zero_grad(set_to_none=True)

    if n_ok == 0:
        print(f"[HF GRAD] Khong co anh nao khop GT/HF mask (gt_dir={gt_dir}, hf_masks_dir={hf_masks_dir})", flush=True)
        return

    avg_grads = [v / n_ok for v in grads_sum]
    avg_contribs = [v / n_ok for v in contribs_sum]
    header = ("iter,Q1_grad,Q2_grad,Q3_grad,Q4_grad,Q5_grad,"
              "Q1_contrib,Q2_contrib,Q3_contrib,Q4_contrib,Q5_contrib")
    _write_csv_row(hf_grad_log_csv, header, iteration, avg_grads, avg_contribs)

    print(f"[HF GRAD] iter={iteration} n={n_ok} Q1_grad={avg_grads[0]:.6f} Q5_grad={avg_grads[4]:.6f}", flush=True)


# VAR: add cap max
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, cap_max=-1, analyse_path=None, orig_dir = None, test_render_every=50, test_render_samples=15, lambda_lpips=0.15, hf_masks_dir=None, hf_log_csv="/kaggle/working/HFlog.csv", hf_eval_every=50, hf_grad_log_csv="/kaggle/working/HF_gradient_log.csv", hf_grad_eval_every=50, hf_grad_samples=20):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim ) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0
        

        loss.backward()

        iter_end.record()

        with torch.no_grad():

            # VAR: HF quintile LOSS evaluation trigger (khong can grad)
            if orig_dir is not None and iteration % hf_eval_every == 0:
                evaluate_hf(
                    dataset, gaussians, pipe, background, iteration,
                    orig_dir=orig_dir, hf_masks_dir=hf_masks_dir, hf_log_csv=hf_log_csv,
                )

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                # VAR: log gaussians numbers 
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}", "N": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # VAR: add capmax constraint 
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if cap_max <= 0 or gaussians.get_xyz.shape[0] < cap_max:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    else:
                        gaussians.tmp_radii = radii
                        prune_mask = (gaussians.get_opacity < 0.005).squeeze()
                        gaussians.prune_points(prune_mask)
                        gaussians.tmp_radii = None
                        torch.cuda.empty_cache()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # VAR: HF quintile GRADIENT evaluation trigger (CAN grad, nen dat
        # NGOAI khoi torch.no_grad(), va SAU khi optimizer.step()+zero_grad()
        # cua iteration nay da xong, de khong lam hong buoc train tiep theo.
        if orig_dir is not None and iteration % hf_grad_eval_every == 0:
            evaluate_hf_gradient(
                dataset, gaussians, pipe, background, iteration,
                orig_dir=orig_dir, hf_masks_dir=hf_masks_dir,
                hf_grad_log_csv=hf_grad_log_csv, num_samples=hf_grad_samples,
            )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--cap_max", type=int, default=-1)
    parser.add_argument("--analyse", type=str, default=None, help="Đường dẫn file log điểm số. Không truyền = không log.")
    parser.add_argument("--orig_dir", type=str, default=None,
        help="Duong dan chua cameras.bin SIMPLE_RADIAL goc (truoc undistort), can de redistort anh test.")
    parser.add_argument("--test_render_every", type=int, default=50)
    parser.add_argument("--test_render_samples", type=int, default=15)
    parser.add_argument("--hf_masks_dir", type=str, default=None,
        help="Duong dan test/hf_masks chua *_highfreq.png (mac dinh: <scene>/test/hf_masks)")
    parser.add_argument("--hf_log_csv", type=str, default="/kaggle/working/HFlog.csv")
    parser.add_argument("--hf_eval_every", type=int, default=50)
    parser.add_argument("--hf_grad_log_csv", type=str, default="/kaggle/working/HF_gradient_log.csv")
    parser.add_argument("--hf_grad_eval_every", type=int, default=50)
    parser.add_argument("--hf_grad_samples", type=int, default=20)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.cap_max, args.analyse, args.orig_dir, args.test_render_every, args.test_render_samples, hf_masks_dir=args.hf_masks_dir, hf_log_csv=args.hf_log_csv, hf_eval_every=args.hf_eval_every, hf_grad_log_csv=args.hf_grad_log_csv, hf_grad_eval_every=args.hf_grad_eval_every, hf_grad_samples=args.hf_grad_samples)
    # All done
    print("\nTraining complete.")