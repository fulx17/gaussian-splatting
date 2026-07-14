import io
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from utils.improved_gs_utils import (
    build_improvedgs_resume_config,
    capture_improvedgs_runtime_state,
    compute_active_gaussian_budget,
    compute_edge_map,
    deterministic_eas_sample_indices,
    erode_alpha_mask,
    mu_update_interval,
    rap_prune_iterations,
    rap_reset_iterations,
    restore_improvedgs_runtime_state,
    seed_everything,
    should_step_optimizer,
    validate_improvedgs_resume_config,
)


class GrowthControlTests(unittest.TestCase):
    def test_square_root_schedule_and_fixed_ablation(self):
        self.assertEqual(
            compute_active_gaussian_budget(500, 500, 15_000, 1_500_000),
            1,
        )
        midpoint = (500 + 14_500) // 2
        self.assertAlmostEqual(
            compute_active_gaussian_budget(
                midpoint, 500, 15_000, 1_500_000
            ),
            int((0.5 ** 0.5) * 1_500_000),
            delta=1,
        )
        self.assertEqual(
            compute_active_gaussian_budget(14_500, 500, 15_000, 1_500_000),
            1_500_000,
        )
        self.assertEqual(
            compute_active_gaussian_budget(
                600, 500, 15_000, 1_500_000, use_growth_control=False
            ),
            1_500_000,
        )


class ScheduleTests(unittest.TestCase):
    def test_mu_paper_boundaries(self):
        self.assertEqual(mu_update_interval(14_999), 1)
        self.assertEqual(mu_update_interval(15_000), 5)
        self.assertEqual(mu_update_interval(22_499), 5)
        self.assertEqual(mu_update_interval(22_500), 20)
        self.assertTrue(should_step_optimizer(15_000, 30_000))
        self.assertFalse(should_step_optimizer(15_001, 30_000))
        self.assertTrue(should_step_optimizer(15_005, 30_000))
        self.assertTrue(should_step_optimizer(22_500, 30_000))
        self.assertTrue(should_step_optimizer(22_520, 30_000))

    def test_resume_safe_eas_rotation(self):
        self.assertEqual(
            deterministic_eas_sample_indices(7, 3, 600, 500, 100),
            [0, 1, 2],
        )
        self.assertEqual(
            deterministic_eas_sample_indices(7, 3, 700, 500, 100),
            [3, 4, 5],
        )
        self.assertEqual(
            deterministic_eas_sample_indices(7, 3, 800, 500, 100),
            [6, 0, 1],
        )

    def test_rap_paper_schedule(self):
        self.assertEqual(
            rap_reset_iterations(500, 15_000, 3_000, 2),
            [3_000, 6_000],
        )
        self.assertEqual(
            rap_prune_iterations(500, 15_000, 3_000, 2, 300),
            [3_300, 6_300],
        )


class EdgeMapTests(unittest.TestCase):
    def test_alpha_erosion_does_not_mutate_input(self):
        alpha = torch.tensor(
            [[2.0, 2.0, 2.0], [2.0, -1.0, 2.0], [2.0, 2.0, 2.0]]
        )
        original = alpha.clone()
        eroded = erode_alpha_mask(alpha, radius=1)
        self.assertTrue(torch.equal(alpha, original))
        self.assertEqual(tuple(eroded.shape), (3, 3))
        self.assertTrue(torch.all((0.0 <= eroded) & (eroded <= 1.0)))

    def test_edge_map_is_cpu_half_and_alpha_masked(self):
        image = torch.zeros((3, 7, 7), dtype=torch.float32)
        image[:, 3, 3] = 1.0
        alpha = torch.ones((1, 7, 7), dtype=torch.float32)
        alpha[:, :2, :] = 0.0
        edge = compute_edge_map(image, alpha_mask=alpha, mask_erosion_radius=1)
        self.assertEqual(edge.device.type, "cpu")
        self.assertEqual(edge.dtype, torch.float16)
        self.assertEqual(tuple(edge.shape), (7, 7))
        self.assertTrue(torch.all(edge[:3] == 0))
        self.assertGreater(float(edge[3, 3]), 0.0)

    def test_seed_controls_torch_sampling(self):
        seed_everything(123)
        first = torch.rand(5)
        seed_everything(123)
        second = torch.rand(5)
        self.assertTrue(torch.equal(first, second))


class CheckpointRuntimeTests(unittest.TestCase):
    class DummyGaussians:
        def __init__(self):
            self.parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
            self.optimizer = torch.optim.Adam(
                [{"params": [self.parameter], "name": "xyz"}], lr=0.01
            )
            self._exposure = torch.nn.Parameter(torch.tensor([[0.5, 0.25]]))
            self.exposure_mapping = {"camera_a": 0}
            self.exposure_optimizer = torch.optim.Adam([self._exposure], lr=0.01)

    def test_pending_gradients_exposure_rng_and_camera_stack_round_trip(self):
        gaussians = self.DummyGaussians()
        # Populate Adam state before recording the pending gradients that MU
        # would carry into a later optimizer boundary.
        gaussians.parameter.grad = torch.ones_like(gaussians.parameter)
        gaussians._exposure.grad = torch.ones_like(gaussians._exposure)
        gaussians.optimizer.step()
        gaussians.exposure_optimizer.step()
        gaussians.parameter.grad = torch.tensor([[2.0, 3.0]])
        gaussians._exposure.grad = torch.tensor([[4.0, 5.0]])
        saved_exposure = gaussians._exposure.detach().clone()

        seed_everything(77)
        runtime_state = capture_improvedgs_runtime_state(
            gaussians,
            [3, 1],
            remaining_camera_names=["camera_d", "camera_b"],
            camera_order_names=["camera_a", "camera_b", "camera_c", "camera_d"],
        )
        # Match a real checkpoint serialization so optimizer-state tensors are
        # frozen rather than sharing Python references with the live optimizer.
        buffer = io.BytesIO()
        torch.save(runtime_state, buffer)
        buffer.seek(0)
        runtime_state = torch.load(buffer, weights_only=False)

        expected_python = random.random()
        expected_numpy = float(np.random.rand())
        expected_torch = torch.rand(3)

        with torch.no_grad():
            gaussians._exposure.fill_(-9.0)
        gaussians.exposure_mapping = {"wrong": 0}
        gaussians.parameter.grad = None
        gaussians._exposure.grad = None
        seed_everything(999)

        restore_improvedgs_runtime_state(gaussians, runtime_state)
        self.assertEqual(runtime_state["viewpoint_indices"], [3, 1])
        self.assertEqual(
            runtime_state["remaining_camera_names"], ["camera_d", "camera_b"]
        )
        self.assertEqual(
            runtime_state["camera_order_names"],
            ["camera_a", "camera_b", "camera_c", "camera_d"],
        )
        self.assertEqual(gaussians.exposure_mapping, {"camera_a": 0})
        self.assertTrue(torch.equal(gaussians._exposure, saved_exposure))
        self.assertTrue(
            torch.equal(gaussians.parameter.grad, torch.tensor([[2.0, 3.0]]))
        )
        self.assertTrue(
            torch.equal(gaussians._exposure.grad, torch.tensor([[4.0, 5.0]]))
        )
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.rand()), expected_numpy)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_resume_configuration_mismatch_is_rejected(self):
        dataset = SimpleNamespace(
            sh_degree=3,
            source_path="/data/scene/train",
            train_test_exp=False,
            white_background=False,
        )
        opt = SimpleNamespace(
            iterations=30_000,
            density_control="improvedgs",
            use_mu=1,
            gaussian_budget=1_500_000,
        )
        pipe = SimpleNamespace(
            antialiasing=False,
            compute_cov3D_python=False,
            convert_SHs_python=False,
        )
        config = build_improvedgs_resume_config(dataset, opt, pipe, seed=7)
        validate_improvedgs_resume_config({"resume_config": config}, config)

        changed_opt = SimpleNamespace(**vars(opt))
        changed_opt.use_mu = 0
        changed_config = build_improvedgs_resume_config(
            dataset, changed_opt, pipe, seed=7
        )
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            validate_improvedgs_resume_config(
                {"resume_config": config}, changed_config
            )

        with self.assertRaisesRegex(ValueError, "pending MU gradients"):
            validate_improvedgs_resume_config(
                {"parameter_grads": {"xyz": torch.ones(1)}},
                changed_config,
            )


if __name__ == "__main__":
    unittest.main()
