"""Small, testable helpers used by the ImprovedGS training path.

This module deliberately has no dependency on the renderer or Gaussian model.
The scheduling helpers can therefore be unit-tested on CPU, while edge maps are
cached as CPU ``float16`` tensors and copied to the GPU only for EAS renders.
"""

from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


_EDGE_KERNEL = torch.tensor(
    [[[-1.0, -1.0, -1.0],
      [-1.0, 8.0, -1.0],
      [-1.0, -1.0, -1.0]]],
    dtype=torch.float32,
).unsqueeze(0)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs with one non-negative integer."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_improvedgs_runtime_state(
    gaussians,
    viewpoint_indices,
    remaining_camera_names=None,
    camera_order_names=None,
) -> dict:
    """Capture non-model state needed for an exact Improved-GS resume.

    ``Parameter.grad`` is not serialized as part of a parameter tensor. Saving
    it explicitly prevents a checkpoint inside a MU accumulation window from
    silently dropping views. The model argument is intentionally duck-typed so
    this function remains independently CPU-testable.
    """
    parameter_grads = {}
    for group in gaussians.optimizer.param_groups:
        if len(group["params"]) != 1:
            raise RuntimeError("Each Gaussian optimizer group must own one tensor")
        grad = group["params"][0].grad
        parameter_grads[group["name"]] = (
            None if grad is None else grad.detach().cpu().clone()
        )

    exposure_grads = []
    for group in gaussians.exposure_optimizer.param_groups:
        for parameter in group["params"]:
            exposure_grads.append(
                None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            )

    return {
        "version": 1,
        "viewpoint_indices": [int(index) for index in viewpoint_indices],
        "remaining_camera_names": (
            None
            if remaining_camera_names is None
            else [str(name) for name in remaining_camera_names]
        ),
        "camera_order_names": (
            None
            if camera_order_names is None
            else [str(name) for name in camera_order_names]
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "parameter_grads": parameter_grads,
        "exposure": gaussians._exposure.detach().cpu().clone(),
        "exposure_mapping": dict(getattr(gaussians, "exposure_mapping", {})),
        "exposure_optimizer": gaussians.exposure_optimizer.state_dict(),
        "exposure_grads": exposure_grads,
    }


def restore_improvedgs_runtime_state(gaussians, runtime_state: dict) -> None:
    """Restore RNG, exposure, optimizer, and pending MU gradients."""
    if int(runtime_state.get("version", -1)) != 1:
        raise ValueError("Unsupported Improved-GS checkpoint runtime-state version")

    exposure = runtime_state["exposure"].to(
        device=gaussians._exposure.device, dtype=gaussians._exposure.dtype
    )
    if exposure.shape != gaussians._exposure.shape:
        raise ValueError("Checkpoint exposure tensor does not match this scene")
    with torch.no_grad():
        gaussians._exposure.copy_(exposure)
    exposure_mapping = runtime_state.get("exposure_mapping")
    if exposure_mapping:
        mapping_values = [int(index) for index in exposure_mapping.values()]
        if (
            len(set(exposure_mapping)) != exposure.shape[0]
            or sorted(mapping_values) != list(range(exposure.shape[0]))
        ):
            raise ValueError("Checkpoint exposure mapping is invalid")
        gaussians.exposure_mapping = {
            str(name): int(index) for name, index in exposure_mapping.items()
        }
    gaussians.exposure_optimizer.load_state_dict(runtime_state["exposure_optimizer"])

    parameter_grads = runtime_state.get("parameter_grads", {})
    expected_names = {group["name"] for group in gaussians.optimizer.param_groups}
    if set(parameter_grads) != expected_names:
        raise ValueError("Checkpoint Gaussian-gradient groups do not match the model")
    for group in gaussians.optimizer.param_groups:
        parameter = group["params"][0]
        saved_grad = parameter_grads[group["name"]]
        parameter.grad = (
            None
            if saved_grad is None
            else saved_grad.to(device=parameter.device, dtype=parameter.dtype)
        )

    exposure_parameters = [
        parameter
        for group in gaussians.exposure_optimizer.param_groups
        for parameter in group["params"]
    ]
    saved_exposure_grads = runtime_state.get("exposure_grads", [])
    if len(saved_exposure_grads) != len(exposure_parameters):
        raise ValueError("Checkpoint exposure-gradient groups do not match the model")
    for parameter, saved_grad in zip(exposure_parameters, saved_exposure_grads):
        parameter.grad = (
            None
            if saved_grad is None
            else saved_grad.to(device=parameter.device, dtype=parameter.dtype)
        )

    random.setstate(runtime_state["python_rng_state"])
    np.random.set_state(runtime_state["numpy_rng_state"])
    torch.set_rng_state(runtime_state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(runtime_state["cuda_rng_state_all"])


def build_improvedgs_resume_config(dataset, opt, pipe, seed: int) -> dict:
    """Return settings that must stay unchanged for a state-complete resume."""
    return {
        "optimization": dict(vars(opt)),
        "pipeline": {
            "antialiasing": bool(pipe.antialiasing),
            "compute_cov3D_python": bool(pipe.compute_cov3D_python),
            "convert_SHs_python": bool(pipe.convert_SHs_python),
        },
        "dataset": {
            "sh_degree": int(dataset.sh_degree),
            "source_path": str(dataset.source_path),
            "train_test_exp": bool(dataset.train_test_exp),
            "white_background": bool(dataset.white_background),
        },
        "seed": int(seed),
    }


def validate_improvedgs_resume_config(runtime_state: dict, current_config: dict) -> None:
    """Reject resume settings that would reinterpret saved pending state."""
    saved_config = runtime_state.get("resume_config")
    if saved_config is None:
        pending_gradients = any(
            gradient is not None
            for gradient in runtime_state.get("parameter_grads", {}).values()
        )
        if pending_gradients and not bool(
            current_config["optimization"].get("use_mu", False)
        ):
            raise ValueError(
                "Checkpoint contains pending MU gradients but the current run "
                "disables MU; refusing to mix incompatible update semantics."
            )
        return
    if saved_config != current_config:
        differing_sections = [
            name
            for name in sorted(set(saved_config) | set(current_config))
            if saved_config.get(name) != current_config.get(name)
        ]
        raise ValueError(
            "Improved-GS checkpoint configuration mismatch in: {}. Resume "
            "with the same seed/method/pipeline settings.".format(
                ", ".join(differing_sections)
            )
        )


def compute_active_gaussian_budget(
    iteration: int,
    densify_from_iter: int,
    densify_until_iter: int,
    final_budget: int,
    use_growth_control: bool = True,
    warmup_until_offset: int = 500,
) -> int:
    """Return the hard Gaussian budget active at ``iteration``.

    With growth control enabled, the budget follows the ImprovedGS square-root
    warm-up and reaches ``final_budget`` shortly before densification ends. With
    it disabled, the same final hard cap is used from the first split, which is
    useful for an ablation that changes only the growth schedule.
    """
    values = (iteration, densify_from_iter, densify_until_iter, final_budget, warmup_until_offset)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("iteration and budget schedule values must be integers")
    if final_budget <= 0:
        raise ValueError("final_budget must be positive")
    if densify_from_iter < 0 or densify_until_iter <= densify_from_iter:
        raise ValueError("densification interval must be positive and ordered")
    if warmup_until_offset < 0:
        raise ValueError("warmup_until_offset must be non-negative")
    if not isinstance(use_growth_control, bool):
        raise TypeError("use_growth_control must be bool")
    if not use_growth_control:
        return final_budget

    warmup_end = densify_until_iter - warmup_until_offset
    if warmup_end <= densify_from_iter or iteration >= warmup_end:
        return final_budget
    progress = (iteration - densify_from_iter) / float(warmup_end - densify_from_iter)
    progress = min(max(progress, 0.0), 1.0)
    return max(int(math.sqrt(progress) * final_budget), 1)


def mu_update_interval(
    iteration: int,
    use_mu: bool = True,
    first_stage_start: int = 15_000,
    second_stage_start: int = 22_500,
    first_stage_interval: int = 5,
    second_stage_interval: int = 20,
) -> int:
    """Return the number of views accumulated per optimizer update.

    The defaults reproduce the paper schedule: update every view before 15k,
    every five views from 15k to 22.5k, and every twenty views afterwards.
    """
    values = (
        iteration,
        first_stage_start,
        second_stage_start,
        first_stage_interval,
        second_stage_interval,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("MU schedule values must be integers")
    if not isinstance(use_mu, bool):
        raise TypeError("use_mu must be bool")
    if iteration < 0 or first_stage_start < 0:
        raise ValueError("MU iterations must be non-negative")
    if second_stage_start <= first_stage_start:
        raise ValueError("second_stage_start must be greater than first_stage_start")
    if first_stage_interval <= 0 or second_stage_interval <= 0:
        raise ValueError("MU intervals must be positive")
    if not use_mu or iteration < first_stage_start:
        return 1
    if iteration < second_stage_start:
        return first_stage_interval
    return second_stage_interval


def should_step_optimizer(
    iteration: int,
    total_iterations: int,
    use_mu: bool = True,
    first_stage_start: int = 15_000,
    second_stage_start: int = 22_500,
    first_stage_interval: int = 5,
    second_stage_interval: int = 20,
) -> bool:
    """Return whether accumulated gradients should be applied this iteration."""
    if isinstance(total_iterations, bool) or not isinstance(total_iterations, int):
        raise TypeError("total_iterations must be an integer")
    if total_iterations <= 0:
        raise ValueError("total_iterations must be positive")
    if iteration >= total_iterations:
        return False
    interval = mu_update_interval(
        iteration,
        use_mu=use_mu,
        first_stage_start=first_stage_start,
        second_stage_start=second_stage_start,
        first_stage_interval=first_stage_interval,
        second_stage_interval=second_stage_interval,
    )
    return interval == 1 or iteration % interval == 0


def erode_alpha_mask(alpha_mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    """Erode a ``[H,W]`` or ``[1,H,W]`` alpha mask without extra packages."""
    if not torch.is_tensor(alpha_mask):
        raise TypeError("alpha_mask must be a torch.Tensor")
    if isinstance(radius, bool) or not isinstance(radius, int):
        raise TypeError("radius must be an integer")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if alpha_mask.ndim == 2:
        mask = alpha_mask.unsqueeze(0).unsqueeze(0)
        output_ndim = 2
    elif alpha_mask.ndim == 3 and alpha_mask.shape[0] == 1:
        mask = alpha_mask.unsqueeze(0)
        output_ndim = 3
    else:
        raise ValueError("alpha_mask must have shape [H,W] or [1,H,W]")
    mask = (
        mask.detach()
        .to(dtype=torch.float32)
        .clone()
        .clamp_(0.0, 1.0)
    )
    if radius:
        kernel_size = radius * 2 + 1
        mask = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=radius)
    mask = mask.squeeze(0)
    return mask.squeeze(0) if output_ndim == 2 else mask


def normalize_to_unit_range(values: torch.Tensor) -> torch.Tensor:
    """Normalize finite tensor values to ``[0,1]``; constants map to zero."""
    if not torch.is_tensor(values):
        raise TypeError("values must be a torch.Tensor")
    sanitized = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if sanitized.numel() == 0:
        return sanitized
    minimum = sanitized.amin()
    scale = sanitized.amax() - minimum
    if float(scale.item()) <= 0.0:
        return torch.zeros_like(sanitized)
    return (sanitized - minimum) / scale


def compute_edge_map(
    image: torch.Tensor,
    alpha_mask: Optional[torch.Tensor] = None,
    mask_erosion_radius: int = 1,
) -> torch.Tensor:
    """Build an ImprovedGS Laplacian edge map cached as CPU ``float16``.

    RGB is converted using integer-style luminance weights from the reference
    implementation. An eroded alpha mask suppresses artificial RGBA/crop edges.
    """
    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")
    if image.ndim != 3 or image.shape[0] < 3:
        raise ValueError("image must have shape [C,H,W] with at least 3 channels")
    rgb = image[:3].detach().to(dtype=torch.float32).unsqueeze(0)
    rgb_255 = torch.round(rgb.clamp(0.0, 1.0) * 255.0)
    grayscale = torch.round(
        (299.0 * rgb_255[:, 0:1] + 587.0 * rgb_255[:, 1:2] + 114.0 * rgb_255[:, 2:3])
        / 1000.0
    )
    response = F.conv2d(
        grayscale,
        _EDGE_KERNEL.to(device=grayscale.device, dtype=grayscale.dtype),
        padding=1,
    )
    edge_map = torch.clamp(response, min=0.0, max=255.0).squeeze(0).squeeze(0) / 255.0
    edge_map = normalize_to_unit_range(edge_map)
    if alpha_mask is not None:
        valid_mask = erode_alpha_mask(alpha_mask, mask_erosion_radius)
        if valid_mask.ndim == 3:
            valid_mask = valid_mask.squeeze(0)
        if valid_mask.shape != edge_map.shape:
            raise ValueError("alpha_mask spatial shape must match image")
        edge_map = edge_map * valid_mask.to(device=edge_map.device)
    return edge_map.to(dtype=torch.float16, device="cpu").contiguous()


def prepare_edge_map_cache(cameras: Sequence[object], mask_erosion_radius: int = 1) -> list[torch.Tensor]:
    """Precompute one CPU-half edge map per camera, preserving camera order."""
    if not isinstance(cameras, Sequence):
        raise TypeError("cameras must be a sequence")
    edge_maps = []
    for index, camera in enumerate(cameras):
        image = getattr(camera, "original_image", None)
        if image is None:
            raise ValueError("camera {} has no original_image".format(index))
        edge_maps.append(
            compute_edge_map(
                image,
                alpha_mask=getattr(camera, "alpha_mask", None),
                mask_erosion_radius=mask_erosion_radius,
            )
        )
    return edge_maps


def rotating_sample_indices(total: int, count: int, cursor: int) -> tuple[list[int], int]:
    """Take deterministic cyclic camera indices and return the next cursor."""
    values = (total, count, cursor)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("sampling arguments must be integers")
    if total <= 0:
        raise ValueError("total must be positive")
    if count == -1 or count >= total:
        return list(range(total)), 0
    if count <= 0:
        raise ValueError("count must be -1 or positive")
    cursor %= total
    indices = [(cursor + offset) % total for offset in range(count)]
    return indices, (cursor + count) % total


def deterministic_eas_sample_indices(
    total_cameras: int,
    sample_count: int,
    iteration: int,
    densify_from_iter: int,
    densification_interval: int,
) -> list[int]:
    """Derive EAS camera indices from the iteration, including after resume.

    No mutable sampling pool is needed. Each densification event advances the
    cyclic start by one sample batch, so restarting at a checkpoint selects the
    same cameras that an uninterrupted run would select for that iteration.
    """
    values = (
        total_cameras,
        sample_count,
        iteration,
        densify_from_iter,
        densification_interval,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("EAS sampling schedule values must be integers")
    if total_cameras <= 0 or densification_interval <= 0:
        raise ValueError("camera count and densification interval must be positive")
    if sample_count == -1 or sample_count >= total_cameras:
        return list(range(total_cameras))
    if sample_count <= 0:
        raise ValueError("sample_count must be -1 or positive")
    first_event = (
        densify_from_iter // densification_interval + 1
    ) * densification_interval
    if iteration < first_event or iteration % densification_interval != 0:
        raise ValueError("iteration must be a densification event after densify_from_iter")
    event_index = (iteration - first_event) // densification_interval
    cursor = (event_index * sample_count) % total_cameras
    indices, _ = rotating_sample_indices(total_cameras, sample_count, cursor)
    return indices


def rap_reset_iterations(
    densify_from_iter: int,
    densify_until_iter: int,
    opacity_reset_interval: int,
    rap_rounds: int,
) -> list[int]:
    """Return the RAP resets that schedule delayed percentile pruning."""
    values = (densify_from_iter, densify_until_iter, opacity_reset_interval, rap_rounds)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("RAP schedule values must be integers")
    if densify_from_iter < 0 or densify_until_iter <= densify_from_iter:
        raise ValueError("densification window must be positive and ordered")
    if opacity_reset_interval <= 0 or rap_rounds < 0:
        raise ValueError("RAP reset interval must be positive and rounds non-negative")
    first_reset = (
        densify_from_iter // opacity_reset_interval + 1
    ) * opacity_reset_interval
    resets = list(range(first_reset, densify_until_iter, opacity_reset_interval))
    return resets[:rap_rounds]


def rap_prune_iterations(
    densify_from_iter: int,
    densify_until_iter: int,
    opacity_reset_interval: int,
    rap_rounds: int,
    rap_prune_offset: int,
) -> list[int]:
    """Return deterministic recovery-aware prune iterations."""
    if isinstance(rap_prune_offset, bool) or not isinstance(rap_prune_offset, int):
        raise TypeError("rap_prune_offset must be an integer")
    if rap_prune_offset < 0:
        raise ValueError("rap_prune_offset must be non-negative")
    return [
        reset_iteration + rap_prune_offset
        for reset_iteration in rap_reset_iterations(
            densify_from_iter,
            densify_until_iter,
            opacity_reset_interval,
            rap_rounds,
        )
    ]
