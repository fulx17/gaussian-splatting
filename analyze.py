"""
Khao sat COLMAP data cho tung scene trong VAR2026:
- Loai camera model (PINHOLE, SIMPLE_RADIAL, ...) va params (fx, fy, cx, cy, k1...)
- So luong points3D
- So luong anh train

Chay:
    python survey_scenes.py --root phase1/private_set1
"""
import os
from argparse import ArgumentParser
from pathlib import Path

from utils.read_write_model import read_cameras_binary, read_points3D_binary, read_images_binary


def find_sparse_dir(scene_dir: Path):
    # cau truc: <scene>/train/sparse/0
    candidates = [
        scene_dir / "train" / "sparse" / "0",
        scene_dir / "sparse" / "0",
    ]
    for c in candidates:
        if (c / "cameras.bin").exists():
            return c
    return None


def survey_scene(scene_name: str, sparse_dir: Path):
    result = {"scene": scene_name}
    try:
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    except Exception as e:
        result["error"] = f"cameras.bin loi: {e}"
        return result

    cam_info = []
    for cam_id, cam in cameras.items():
        cam_info.append({
            "id": cam_id,
            "model": cam.model,
            "width": cam.width,
            "height": cam.height,
            "params": [round(float(p), 6) for p in cam.params],
        })
    result["cameras"] = cam_info

    try:
        points3D = read_points3D_binary(str(sparse_dir / "points3D.bin"))
        result["num_points3D"] = len(points3D)
    except Exception as e:
        result["num_points3D"] = None
        result["points3D_error"] = str(e)

    try:
        images = read_images_binary(str(sparse_dir / "images.bin"))
        result["num_images"] = len(images)
    except Exception as e:
        result["num_images"] = None
        result["images_error"] = str(e)

    return result


def main():
    parser = ArgumentParser()
    parser.add_argument("--root", required=True, help="Thu muc chua cac scene, vd: phase1/private_set1")
    args = parser.parse_args()

    root = Path(args.root)
    scenes = sorted([d for d in root.iterdir() if d.is_dir()])

    all_results = []
    for scene_dir in scenes:
        sparse_dir = find_sparse_dir(scene_dir)
        if sparse_dir is None:
            print(f"[{scene_dir.name}] !!! khong tim thay sparse/0, bo qua")
            continue
        res = survey_scene(scene_dir.name, sparse_dir)
        all_results.append(res)

    # In bang tong hop
    print("\n" + "=" * 100)
    print(f"{'Scene':<12} {'CamModel':<15} {'W x H':<12} {'#Cams':<6} {'#Images':<8} {'#Points3D':<10} {'Params'}")
    print("=" * 100)
    for res in all_results:
        if "error" in res:
            print(f"{res['scene']:<12} LOI: {res['error']}")
            continue
        for cam in res["cameras"]:
            wh_str = f"{cam['width']}x{cam['height']}"
            print(
                f"{res['scene']:<12} "
                f"{cam['model']:<15} "
                f"{wh_str:<12} "
                f"{len(res['cameras']):<6} "
                f"{str(res.get('num_images')):<8} "
                f"{str(res.get('num_points3D')):<10} "
                f"{cam['params']}"
            )

    # Gom nhom theo camera model de de nhin outlier
    print("\n" + "=" * 100)
    print("Gom nhom theo camera model:")
    model_groups = {}
    for res in all_results:
        if "error" in res:
            continue
        for cam in res["cameras"]:
            model_groups.setdefault(cam["model"], []).append(res["scene"])
    for model, scenes_list in model_groups.items():
        print(f"  {model}: {len(scenes_list)} scenes -> {scenes_list}")

    # Sap xep theo so points3D de de thay scene co sparse cloud qua it
    print("\n" + "=" * 100)
    print("Sap xep theo so points3D (tang dan, de phat hien scene thieu sparse points):")
    valid = [r for r in all_results if r.get("num_points3D") is not None]
    valid.sort(key=lambda r: r["num_points3D"])
    for r in valid:
        print(f"  {r['scene']:<12} num_points3D={r['num_points3D']:<10} num_images={r.get('num_images')}")


if __name__ == "__main__":
    main()