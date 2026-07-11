import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from utils.read_write_model import (
    read_images_binary,
    read_cameras_binary,
    read_points3D_binary,
    write_images_binary,
    write_cameras_binary,
    write_points3D_binary,
)

def preprocess_scene(scene_path):

    scene_path = Path(scene_path)

    images_dir = scene_path / "train" / "images"
    sparse_path = scene_path / "train" / "sparse" / "0"

    cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))
    images = read_images_binary(str(sparse_path / "images.bin"))
    points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))

    existing_files = set(os.listdir(images_dir))

    # dùng set để tra cứu O(1)
    missing_ids = {
        img_id
        for img_id, img in images.items()
        if img.name not in existing_files
    }

    print(f"scene: {scene_path.name}")
    print(f"Tổng ảnh: {len(images)}")
    print(f"Số ảnh missing: {len(missing_ids)}")

    for img_id in missing_ids:
        del images[img_id]

    print(f"Còn lại sau khi xóa: {len(images)}")
    # ghi đè luôn
    write_cameras_binary(cameras, str(sparse_path / "cameras.bin"))
    write_images_binary(images, str(sparse_path / "images.bin"))
    write_points3D_binary(points3D, str(sparse_path / "points3D.bin"))

    print(f"Đã cập nhật reconstruction.")

def undistort_scene(scene_path, blank_pixels=1.0, min_scale=1.0, max_scale=2.0):
    """
    Undistort dung COLMAP CLI (C++) - nhanh hon nhieu so voi pycolmap python API,
    vi chay truc tiep binary C++, khong qua overhead Python/GIL.

    blank_pixels=1.0: chap nhan toi da pixel den, uu tien KHONG CROP mat noi dung goc
                       (tuong duong --blank_pixels 1 trong docs COLMAP).
    min_scale/max_scale: khoang scale COLMAP duoc phep tu dieu chinh de thoa blank_pixels.

    LUU Y: COLMAP tu dong chon scale NHO NHAT du de thoa dieu kien blank_pixels,
    khong co nghia la no se dung het max_scale. Neu can canvas lon hon, phai
    tu can thiep them (vd dung script CV2 thu cong da test o cho khac).
    """
    scene_path = Path(scene_path)

    image_dir = scene_path / "train" / "images"
    sparse_dir = scene_path / "train" / "sparse" / "0"

    tmp_dir = Path(tempfile.mkdtemp(prefix="undistort_"))

    print(f"Undistorting {scene_path.name} (COLMAP CLI)...")

    try:
        result = subprocess.run(
            [
                "colmap", "image_undistorter",
                "--image_path", str(image_dir),
                "--input_path", str(sparse_dir),
                "--output_path", str(tmp_dir),
                "--output_type", "COLMAP",
                "--blank_pixels", str(blank_pixels),
                "--min_scale", str(min_scale),
                "--max_scale", str(max_scale),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Thay ảnh
        shutil.rmtree(image_dir)
        shutil.copytree(tmp_dir / "images", image_dir)

        # Thay sparse
        shutil.rmtree(sparse_dir)
        shutil.copytree(tmp_dir / "sparse", sparse_dir)

    except subprocess.CalledProcessError as e:
        print(f"LOI khi undistort {scene_path.name}:")
        print(e.stderr)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def preprocess_dataset(path, output_dir):
    path = Path(path)
    output_dir = Path(output_dir)

    shutil.copytree(path, output_dir, dirs_exist_ok=True)

    for scene in os.listdir(path):
        output_scene_path = output_dir / scene 
        undistort_scene(output_scene_path)     
        preprocess_scene(output_scene_path)  

def validate(path):
    path = Path(path)
    for scene in os.listdir(path):
        scene_path = path / scene
        images_dir = scene_path / "train" / "images"
        sparse_path = scene_path / "train" / "sparse" / "0"

        cameras = read_cameras_binary(str(sparse_path / "cameras.bin"))
        images = read_images_binary(str(sparse_path / "images.bin"))
        points3D = read_points3D_binary(str(sparse_path / "points3D.bin"))

        on_disk = set(os.listdir(images_dir))
        registered = {img.name for img in images.values()}

        missing = registered - on_disk
        extra = on_disk - registered
        non_pinhole = [c for c in cameras.values() if c.model != "PINHOLE"]

        print(f"scene: {scene}")
        print(f"  images (sparse/disk): {len(images)}/{len(on_disk)}")
        print(f"  points3D: {len(points3D)}")
        if missing:
            print(f"  ⚠ missing on disk: {len(missing)} e.g. {list(missing)[:3]}")
        if extra:
            print(f"  ⚠ extra on disk (not in sparse): {len(extra)}")
        if non_pinhole:
            print(f"  ⚠ non-PINHOLE cameras (undistort may have failed): {len(non_pinhole)}")
        if not missing and not extra and not non_pinhole:
            print("  ✓ OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess COLMAP dataset by removing missing images and updating sparse model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="/kaggle/working/cleaned_inputs"
    )

    args = parser.parse_args()

    preprocess_dataset(args.input, args.output)
    # validate(args.output)

# preprocess_dataset(r"C:\contest\VAR2026\phase1\private_set1", r"C:\contest\VAR2026\dataset")