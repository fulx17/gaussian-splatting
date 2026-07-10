import os
import sys
import subprocess
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    parser = ArgumentParser(description="Render all scenes with trained 3DGS")
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--image_dir", default="/kaggle/working/image_outputs")
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--subset", nargs="+", default=[])
    parser.add_argument("--extra_args", nargs="*", default=[])
    args = parser.parse_args()

    scenes = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print(f"Found {len(scenes)} scenes: {scenes}")

    if args.subset:
        missing = [s for s in args.subset if s not in scenes]
        if missing:
            print(f"!!! Warning: subset scenes not found in input_dir: {missing}")
        scenes = [s for s in scenes if s in args.subset]
        print(f"Filtered to subset ({len(scenes)}): {scenes}")

    failed_scenes = []

    for i, scene in enumerate(scenes):
        scene_out = Path(args.image_dir) / scene
        if args.skip_existing and scene_out.exists() and any(scene_out.iterdir()):
            print(f"[{i+1}/{len(scenes)}] Skip {scene} (already rendered)")
            continue

        model_path = os.path.join(args.model_dir, scene)
        cfg_path = os.path.join(model_path, "cfg_args")
        if not os.path.exists(cfg_path):
            print(f"!!! Skip {scene}: cfg_args not found ({cfg_path}), scene chưa train xong")
            failed_scenes.append(scene)
            continue

        print(f"\n=== [{i+1}/{len(scenes)}] Rendering scene: {scene} ===")
        cmd = [
            sys.executable, "render_scene.py",
            "--model_dir", args.model_dir,
            "--input_dir", args.input_dir,
            "--image_dir", args.image_dir,
            "--scene_name", scene,
            "--iteration", str(args.iteration),
        ]
        if args.quiet:
            cmd.append("--quiet")
        cmd += args.extra_args

        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"!!! Scene {scene} failed (code {ret.returncode}), continuing...")
            failed_scenes.append(scene)

    print(f"\nDone. {len(scenes) - len(failed_scenes)}/{len(scenes)} scenes succeeded.")
    if failed_scenes:
        print(f"Failed scenes: {failed_scenes}")