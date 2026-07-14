import os
import sys
from argparse import ArgumentParser

from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.improved_gs_utils import seed_everything
from train import training  # điều chỉnh import cho đúng nơi định nghĩa hàm training()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training VAR scene")
    lp = ModelParams(parser)          # không dùng sentinel -> giữ default thật (sh_degree=3, resolution=-1,...)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--input_dir", default="/kaggle/working/cleaned_inputs")
    parser.add_argument("--model_dir", default="/kaggle/working/model_outputs")
    parser.add_argument("--scene_name", required=True)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--cap_max", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    # Tự động ghép source_path từ input_dir + scene_name + train nếu chưa truyền tay
    if getattr(args, "source_path", None) in (None, ""):
        input_dir = getattr(args, "input_dir", None)
        scene = getattr(args, "scene_name", None)
        if input_dir is not None and scene is not None:
            args.source_path = os.path.join(input_dir, scene, "train")

    # Tự động ghép model_path từ model_dir + scene_name nếu chưa truyền tay
    if getattr(args, "model_path", None) in (None, ""):
        model_dir = getattr(args, "model_dir", None)
        scene = getattr(args, "scene_name", None)
        if model_dir is not None and scene is not None:
            args.model_path = os.path.join(model_dir, scene)

    os.makedirs(args.model_path, exist_ok=True)

    args.save_iterations.append(args.iterations)

    # train_scene imports training() directly, so train.py's __main__ RNG
    # initialization is not executed here.
    safe_state(args.quiet)
    seed_everything(args.seed)

    print(f"Training scene: {args.scene_name}")
    print(f"  source_path = {args.source_path}")
    print(f"  model_path  = {args.model_path}")
    print(f"  density     = {args.density_control} (seed={args.seed})")

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.cap_max,
        seed=args.seed,
    )
