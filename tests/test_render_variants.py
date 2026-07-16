import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, JpegImagePlugin

from utils.image_utils import save_render_jpeg


ROOT = Path(__file__).resolve().parents[1]


def _load_render_helpers():
    """Load CPU-only render helpers without importing the CUDA rasterizer."""
    source = (ROOT / "render_scene.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    names = {
        "_validate_interpolation",
        "build_redistortion_grid",
        "redistort_image",
        "unsharp_mask",
        "build_render_variant_specs",
    }
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"math": __import__("math"), "torch": torch, "F": F}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "render_scene.py", "exec"),
        namespace,
    )
    return namespace


HELPERS = _load_render_helpers()
BUILD_GRID = HELPERS["build_redistortion_grid"]
REDISTORT = HELPERS["redistort_image"]
UNSHARP = HELPERS["unsharp_mask"]
VARIANT_SPECS = HELPERS["build_render_variant_specs"]


class RenderVariantTests(unittest.TestCase):
    def test_variant_mode_builds_the_four_expected_outputs(self):
        self.assertEqual(
            VARIANT_SPECS(True),
            (
                ("bilinear_sharp0", "bilinear", 0.0),
                ("bilinear_sharp0p3", "bilinear", 0.3),
                ("bicubic_sharp0", "bicubic", 0.0),
                ("bicubic_sharp0p3", "bicubic", 0.3),
            ),
        )

    def test_single_mode_preserves_legacy_defaults(self):
        self.assertEqual(
            VARIANT_SPECS(False),
            ((None, "bilinear", 0.0),),
        )

    def test_invalid_interpolation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bilinear.*bicubic"):
            VARIANT_SPECS(False, redistort_interpolation="nearest")

    def test_zero_distortion_is_identity_for_both_interpolations(self):
        image = torch.linspace(0.0, 1.0, 3 * 9 * 11).reshape(3, 9, 11)
        grid = BUILD_GRID(
            9,
            11,
            f=8.0,
            cx=5.0,
            cy=4.0,
            k=0.0,
            device=image.device,
        )
        for interpolation in ("bilinear", "bicubic"):
            with self.subTest(interpolation=interpolation):
                output = REDISTORT(
                    image,
                    f=8.0,
                    cx=5.0,
                    cy=4.0,
                    k=0.0,
                    interpolation=interpolation,
                    grid=grid,
                )
                self.assertTrue(torch.allclose(output, image, atol=2e-6, rtol=0.0))

    def test_precomputed_grid_matches_implicit_grid(self):
        image = torch.rand((3, 10, 12), generator=torch.Generator().manual_seed(7))
        grid = BUILD_GRID(
            10,
            12,
            f=9.0,
            cx=5.5,
            cy=4.5,
            k=0.03,
            device=image.device,
        )
        expected = REDISTORT(
            image,
            f=9.0,
            cx=5.5,
            cy=4.5,
            k=0.03,
            interpolation="bicubic",
        )
        actual = REDISTORT(
            image,
            f=9.0,
            cx=5.5,
            cy=4.5,
            k=0.03,
            interpolation="bicubic",
            grid=grid,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_unsharp_zero_is_noop_and_constant_image_is_stable(self):
        image = torch.full((3, 9, 9), 0.4)
        self.assertIs(UNSHARP(image, amount=0.0), image)
        sharpened = UNSHARP(image, amount=0.3, sigma=0.7)
        self.assertTrue(torch.allclose(sharpened, image, atol=1e-6, rtol=0.0))

    def test_unsharp_increases_an_unsaturated_impulse(self):
        image = torch.zeros((3, 9, 9))
        image[:, 4, 4] = 0.5
        sharpened = UNSHARP(image, amount=0.3, sigma=0.7)
        self.assertTrue(torch.all(sharpened[:, 4, 4] > image[:, 4, 4]))
        self.assertGreaterEqual(float(sharpened.min()), 0.0)
        self.assertLessEqual(float(sharpened.max()), 1.0)

    def test_jpeg_444_and_optimized_encoding(self):
        image = torch.rand((3, 32, 40), generator=torch.Generator().manual_seed(3))
        with tempfile.TemporaryDirectory() as directory:
            plain_path = Path(directory) / "plain.jpg"
            optimized_path = Path(directory) / "optimized.jpg"
            save_render_jpeg(
                image,
                plain_path,
                quality=96,
                subsampling=0,
                optimize=False,
            )
            save_render_jpeg(
                image,
                optimized_path,
                quality=96,
                subsampling=0,
                optimize=True,
            )

            with Image.open(optimized_path) as encoded:
                self.assertEqual(JpegImagePlugin.get_sampling(encoded), 0)
                optimized_pixels = np.asarray(encoded.convert("RGB"))
            with Image.open(plain_path) as encoded:
                plain_pixels = np.asarray(encoded.convert("RGB"))

            self.assertTrue(np.array_equal(plain_pixels, optimized_pixels))

    def test_invalid_jpeg_subsampling_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "subsampling"):
                save_render_jpeg(
                    torch.zeros((3, 4, 4)),
                    Path(directory) / "bad.jpg",
                    subsampling=3,
                )


if __name__ == "__main__":
    unittest.main()
