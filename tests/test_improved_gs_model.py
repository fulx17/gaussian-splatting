import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_gaussian_model_module():
    # The algorithmic tests run on CPU and do not need PLY I/O or simple-knn's
    # CUDA extension. Stub only those optional import-time dependencies.
    if "plyfile" not in sys.modules:
        plyfile = types.ModuleType("plyfile")
        plyfile.PlyData = object
        plyfile.PlyElement = object
        sys.modules["plyfile"] = plyfile

    if "simple_knn._C" not in sys.modules:
        simple_knn = types.ModuleType("simple_knn")
        simple_knn.__path__ = []
        simple_knn_c = types.ModuleType("simple_knn._C")
        simple_knn_c.distCUDA2 = lambda tensor: torch.zeros(
            tensor.shape[0], dtype=tensor.dtype, device=tensor.device
        )
        sys.modules["simple_knn"] = simple_knn
        sys.modules["simple_knn._C"] = simple_knn_c

    spec = importlib.util.spec_from_file_location(
        "gaussian_model_under_test", ROOT / "scene" / "gaussian_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GM = _load_gaussian_model_module()


def _identity_rotations(quaternions):
    return torch.eye(
        3, dtype=quaternions.dtype, device=quaternions.device
    ).unsqueeze(0).repeat(quaternions.shape[0], 1, 1)


def _make_model(scales, opacities=None):
    scales = torch.as_tensor(scales, dtype=torch.float32)
    count = scales.shape[0]
    if opacities is None:
        opacities = torch.full((count, 1), 0.5, dtype=torch.float32)
    else:
        opacities = torch.as_tensor(opacities, dtype=torch.float32).reshape(count, 1)

    model = GM.GaussianModel(sh_degree=0)
    model._xyz = torch.nn.Parameter(
        torch.stack(
            (torch.arange(count, dtype=torch.float32) * 10.0,
             torch.zeros(count),
             torch.zeros(count)),
            dim=1,
        )
    )
    model._features_dc = torch.nn.Parameter(torch.zeros((count, 1, 3)))
    model._features_rest = torch.nn.Parameter(torch.zeros((count, 0, 3)))
    model._scaling = torch.nn.Parameter(torch.log(scales))
    model._rotation = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
    )
    model._opacity = torch.nn.Parameter(GM.inverse_sigmoid(opacities))

    groups = [
        {"params": [model._xyz], "name": "xyz"},
        {"params": [model._features_dc], "name": "f_dc"},
        {"params": [model._features_rest], "name": "f_rest"},
        {"params": [model._opacity], "name": "opacity"},
        {"params": [model._scaling], "name": "scaling"},
        {"params": [model._rotation], "name": "rotation"},
    ]
    model.optimizer = torch.optim.Adam(groups, lr=0.0)
    model.max_radii2D = torch.zeros(count)
    model.xyz_gradient_accum = torch.zeros((count, 1))
    model.xyz_gradient_accum_abs = torch.zeros((count, 1))
    model.denom = torch.zeros((count, 1))
    model.tmp_radii = None
    return model


class LongAxisSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_build_rotation = GM.build_rotation
        GM.build_rotation = _identity_rotations

    @classmethod
    def tearDownClass(cls):
        GM.build_rotation = cls.original_build_rotation

    def test_las_geometry_scale_opacity_and_net_growth(self):
        model = _make_model([[2.0, 1.0, 0.5], [1.0, 3.0, 1.0]], [0.5, 0.25])
        split = model._long_axis_split(
            torch.tensor([True, False]),
            split_distance=0.45,
            opacity_reduction=0.6,
        )

        self.assertEqual(split, 1)
        self.assertEqual(model.get_xyz.shape[0], 3)
        # The unsplit second parent remains first; children follow at +/- 3*rho*sigma_max.
        self.assertTrue(torch.allclose(model.get_xyz[0], torch.tensor([10.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(model.get_xyz[1], torch.tensor([2.7, 0.0, 0.0])))
        self.assertTrue(torch.allclose(model.get_xyz[2], torch.tensor([-2.7, 0.0, 0.0])))

        short_factor = (1.0 - 0.45 ** 2) ** 0.5
        expected_child_scale = torch.tensor(
            [2.0 * (1.0 - 0.45), 1.0 * short_factor, 0.5 * short_factor]
        )
        self.assertTrue(torch.allclose(model.get_scaling[1], expected_child_scale))
        self.assertTrue(torch.allclose(model.get_scaling[2], expected_child_scale))
        self.assertTrue(
            torch.allclose(model.get_opacity[1:], torch.full((2, 1), 0.3))
        )

    def test_hard_budget_limits_parent_count(self):
        model = _make_model([[1.0, 0.8, 0.5]] * 3)
        selected = model.densify_and_split_improved(
            grad_values=torch.ones(3),
            grad_threshold=0.0003,
            budget=4,
            sampling_weights=None,
        )
        self.assertEqual(selected, 1)
        self.assertEqual(model.get_xyz.shape[0], 4)

    def test_eas_samples_positive_weights_only(self):
        model = _make_model([[1.0, 0.8, 0.5]] * 3)
        selected = model.densify_and_split_improved(
            grad_values=torch.ones(3),
            grad_threshold=0.0003,
            budget=6,
            sampling_weights=torch.tensor([1.0, 0.0, 0.0]),
        )
        self.assertEqual(selected, 1)
        self.assertEqual(model.get_xyz.shape[0], 4)

        zero_model = _make_model([[1.0, 0.8, 0.5]] * 3)
        selected = zero_model.densify_and_split_improved(
            grad_values=torch.ones(3),
            grad_threshold=0.0003,
            budget=6,
            sampling_weights=torch.zeros(3),
        )
        self.assertEqual(selected, 0)
        self.assertEqual(zero_model.get_xyz.shape[0], 3)


class ModelStateTests(unittest.TestCase):
    def test_baseline_checkpoint_shape_is_preserved(self):
        model = _make_model([[1.0, 1.0, 1.0]])
        model.xyz_gradient_accum_abs = None
        self.assertEqual(len(model.capture()), 12)
        model.xyz_gradient_accum_abs = torch.zeros((1, 1))
        self.assertEqual(len(model.capture()), 13)

    def test_exact_percent_prune(self):
        model = _make_model(
            [[1.0, 1.0, 1.0]] * 5,
            [0.1, 0.2, 0.3, 0.4, 0.5],
        )
        pruned = model.only_prune(0.4, percent=True)
        self.assertEqual(pruned, 2)
        self.assertEqual(model.get_xyz.shape[0], 3)
        self.assertTrue(torch.all(model.get_opacity >= 0.3 - 1e-6))

    def test_opacity_reset_is_safe_before_adam_state_exists(self):
        model = _make_model([[1.0, 1.0, 1.0]], [0.5])
        self.assertEqual(len(model.optimizer.state), 0)
        model.reset_opacity(0.05)
        self.assertLessEqual(float(model.get_opacity.max()), 0.05 + 1e-6)
        self.assertEqual(len(model.optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
