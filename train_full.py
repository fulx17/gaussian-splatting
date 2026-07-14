import os
import sys
import subprocess
from argparse import ArgumentParser, REMAINDER

if __name__ == "__main__":
    parser = ArgumentParser(description="Train all scenes")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30000])
    parser.add_argument(
        "--extra_args",
        nargs=REMAINDER,
        default=[],
        help="arguments forwarded verbatim to train_scene.py; this option must be last",
    )
    parser.add_argument("--subset", nargs="+", default=[])
    parser.add_argument("--cap_max", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density_control", choices=("3dgs", "improvedgs"), default="3dgs")
    parser.add_argument("--gaussian_budget", type=int, default=1_500_000)
    parser.add_argument("--use_las", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use_rap", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use_gc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use_absgrad", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use_eas", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use_mu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--improvedgs_grad_threshold", type=float, default=0.0003)
    parser.add_argument("--min_opacity", type=float, default=0.005)
    parser.add_argument("--split_distance", type=float, default=0.45)
    parser.add_argument("--opacity_reduction", type=float, default=0.6)
    parser.add_argument("--budget_warmup_until_offset", type=int, default=500)
    parser.add_argument("--improvedgs_reset_max_opacity", type=float, default=0.05)
    parser.add_argument("--rap_initial_prune", type=int, choices=(0, 1), default=1)
    parser.add_argument("--rap_initial_prune_iter", type=int, default=300)
    parser.add_argument("--rap_initial_prune_opacity", type=float, default=0.02)
    parser.add_argument("--rap_prune_ratio", type=float, default=0.20)
    parser.add_argument("--rap_prune_offset", type=int, default=300)
    parser.add_argument("--rap_rounds", type=int, default=2)
    parser.add_argument("--edge_sample_cams", type=int, default=10)
    parser.add_argument("--edge_mask_erosion", type=int, default=1)
    parser.add_argument("--mu_start_iter", type=int, default=15_000)
    parser.add_argument("--mu_interval", type=int, default=5)
    parser.add_argument("--mu_second_start_iter", type=int, default=22_500)
    parser.add_argument("--mu_second_interval", type=int, default=20)
    args = parser.parse_args()

    # sort 
    scenes = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print(f"Found {len(scenes)} scenes: {scenes}")

    # using subset scene if exsist
    if args.subset:
        missing = [s for s in args.subset if s not in scenes]
        if missing:
            print(f"!!! Warning: subset scenes not found in input_dir: {missing}")
        scenes = [s for s in scenes if s in args.subset]
        print(f"Filtered to subset ({len(scenes)}): {scenes}")

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    failed_scenes = []

    for i, scene in enumerate(scenes):
        model_path = os.path.join(args.model_dir, scene)

        print(f"\n=== [{i+1}/{len(scenes)}] Training scene: {scene} ===")
        cmd = [
            sys.executable, "train_scene.py",
            "--input_dir", args.input_dir,
            "--model_dir", args.model_dir,
            "--scene_name", scene,
            "--iterations", str(args.iterations),
            "--test_iterations", *map(str, args.test_iterations),
            "--save_iterations", *map(str, args.save_iterations),
            "--cap_max", str(args.cap_max),
            "--seed", str(args.seed),
            "--density_control", args.density_control,
        ]

        if args.density_control == "improvedgs":
            cmd.extend([
                "--gaussian_budget", str(args.gaussian_budget),
                "--use_las", str(args.use_las),
                "--use_rap", str(args.use_rap),
                "--use_gc", str(args.use_gc),
                "--use_absgrad", str(args.use_absgrad),
                "--use_eas", str(args.use_eas),
                "--use_mu", str(args.use_mu),
                "--improvedgs_grad_threshold", str(args.improvedgs_grad_threshold),
                "--min_opacity", str(args.min_opacity),
                "--split_distance", str(args.split_distance),
                "--opacity_reduction", str(args.opacity_reduction),
                "--budget_warmup_until_offset", str(args.budget_warmup_until_offset),
                "--improvedgs_reset_max_opacity", str(args.improvedgs_reset_max_opacity),
                "--rap_initial_prune", str(args.rap_initial_prune),
                "--rap_initial_prune_iter", str(args.rap_initial_prune_iter),
                "--rap_initial_prune_opacity", str(args.rap_initial_prune_opacity),
                "--rap_prune_ratio", str(args.rap_prune_ratio),
                "--rap_prune_offset", str(args.rap_prune_offset),
                "--rap_rounds", str(args.rap_rounds),
                "--edge_sample_cams", str(args.edge_sample_cams),
                "--edge_mask_erosion", str(args.edge_mask_erosion),
                "--mu_start_iter", str(args.mu_start_iter),
                "--mu_interval", str(args.mu_interval),
                "--mu_second_start_iter", str(args.mu_second_start_iter),
                "--mu_second_interval", str(args.mu_second_interval),
            ])

        # Keep the escape hatch last so an explicitly supplied option wins.
        cmd.extend(args.extra_args)

        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"!!! Scene {scene} failed (code {ret.returncode}), continuing...")
            failed_scenes.append(scene)

    print(f"\nDone. {len(scenes) - len(failed_scenes)}/{len(scenes)} scenes succeeded.")
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")
