import unittest
from argparse import ArgumentParser

from arguments import OptimizationParams


class ImprovedGsCliTests(unittest.TestCase):
    def test_baseline_remains_default(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        parsed = group.extract(parser.parse_args([]))
        self.assertEqual(parsed.density_control, "3dgs")
        self.assertEqual(parsed.gaussian_budget, 1_500_000)

    def test_component_ablation_switches_accept_zero(self):
        parser = ArgumentParser()
        group = OptimizationParams(parser)
        args = parser.parse_args(
            [
                "--density_control", "improvedgs",
                "--use_las", "0",
                "--use_rap", "0",
                "--use_gc", "0",
                "--use_absgrad", "0",
                "--use_eas", "0",
                "--use_mu", "0",
                "--gaussian_budget", "1234",
            ]
        )
        parsed = group.extract(args)
        self.assertEqual(parsed.density_control, "improvedgs")
        self.assertEqual(parsed.gaussian_budget, 1234)
        for name in (
            "use_las", "use_rap", "use_gc", "use_absgrad", "use_eas", "use_mu"
        ):
            self.assertEqual(getattr(parsed, name), 0)


if __name__ == "__main__":
    unittest.main()
