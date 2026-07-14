"""Tiny GPU forward/backward smoke test for the patched rasterizer."""

from __future__ import annotations

import torch

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def _render(settings, means2d):
    device = means2d.device
    rasterizer = GaussianRasterizer(raster_settings=settings)
    return rasterizer(
        means3D=torch.tensor([[0.0, 0.0, 2.0]], device=device),
        means2D=means2d,
        colors_precomp=torch.tensor([[0.8, 0.4, 0.2]], device=device),
        opacities=torch.tensor([[0.8]], device=device),
        scales=torch.tensor([[0.35, 0.25, 0.20]], device=device),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA-enabled PyTorch build")

    device = torch.device("cuda")
    height = width = 32
    common = dict(
        image_height=height,
        image_width=width,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device),
        projmatrix=torch.eye(4, device=device),
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )

    # Legacy callers still use an (N,3) screen-space dummy and expect three
    # public outputs plus a zero third gradient component.
    legacy_means2d = torch.zeros((1, 3), device=device, requires_grad=True)
    baseline_outputs = _render(
        GaussianRasterizationSettings(**common, pixel_weights=None),
        legacy_means2d,
    )
    if len(baseline_outputs) != 3:
        raise AssertionError("Baseline rasterizer API must return three outputs")
    baseline_outputs[0].sum().backward()
    if legacy_means2d.grad is None or float(legacy_means2d.grad[0, 2]) != 0.0:
        raise AssertionError("Legacy means2D third gradient must remain zero")

    # EAS requests a fourth output and AbsGrad writes positive absolute
    # contributions into screen-gradient channels 2-3.
    means2d = torch.zeros((1, 4), device=device, requires_grad=True)
    pixel_weights = torch.zeros((height, width), device=device)
    pixel_weights[:, width // 2 :] = 1.0
    image, radii, depth, accum_weights = _render(
        GaussianRasterizationSettings(**common, pixel_weights=pixel_weights),
        means2d,
    )
    image.sum().backward()

    if radii.numel() != 1 or int(radii[0]) <= 0:
        raise AssertionError("Smoke Gaussian was not visible")
    if accum_weights.shape != (1,) or not torch.isfinite(accum_weights).all():
        raise AssertionError("EAS must return one finite score per Gaussian")
    if float(accum_weights[0]) <= 0.0:
        raise AssertionError("EAS score should be positive for weighted pixels")
    if means2d.grad is None or not torch.isfinite(means2d.grad).all():
        raise AssertionError("AbsGrad screen gradients are missing or non-finite")
    if float(means2d.grad[0, 2:].sum()) <= 0.0:
        raise AssertionError("AbsGrad channels did not accumulate positive gradients")
    if not torch.isfinite(image).all() or not torch.isfinite(depth).all():
        raise AssertionError("Rasterizer produced non-finite image/depth values")

    print(
        "Improved-GS rasterizer smoke test passed on {} (AbsGrad={}, EAS={:.6g}).".format(
            torch.cuda.get_device_name(0),
            means2d.grad[0, 2:].detach().cpu().tolist(),
            float(accum_weights[0]),
        )
    )


if __name__ == "__main__":
    main()
