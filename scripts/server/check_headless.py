"""Render and decode a short synthetic animation without model weights or MANO.

Run from the repository root with its Python environment and DISPLAY unset:
    python scripts/server/check_headless.py --output-dir /tmp/hawor-render-smoke
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lib.vis.run_vis2 import run_vis2_on_video, run_vis2_on_video_cam


WIDTH, HEIGHT, FRAMES, FPS = 320, 240, 6, 30


def synthetic_mesh(center, direction):
    vertices = np.array(
        [[-0.09, -0.08, 0], [0.09, -0.08, 0], [0, 0.09, 0], [0, 0, 0.16]],
        dtype=np.float32,
    )
    trajectory = np.linspace(-0.06, 0.06, FRAMES, dtype=np.float32)
    animation = vertices[None] + np.array(center, dtype=np.float32)
    animation = np.repeat(animation, FRAMES, axis=0)
    animation[:, :, direction] += trajectory[:, None]
    return {
        "vertices": torch.from_numpy(animation[None]),
        "faces": np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
    }


def check_video(path, expect_background=False):
    assert path is not None and Path(path).is_file(), f"Missing returned video: {path}"
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), f"Cannot decode {path}"
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert len(frames) == FRAMES, f"Expected {FRAMES} frames, got {len(frames)}"
    assert frames[0].shape[:2] == (HEIGHT, WIDTH), frames[0].shape
    assert abs(fps - FPS) < 0.01, f"Expected {FPS} fps, got {fps}"
    if expect_background:
        assert np.abs(frames[0][12, 12].astype(float) - 24).max() < 8, "Image background missing"
    first = frames[0].astype(np.int16)
    blue, green, red = first.transpose(2, 0, 1)
    assert np.count_nonzero((blue > red + 30) & (blue > green + 10)) > 10, "Blue mesh missing"
    assert np.count_nonzero((red > green + 15) & (blue > green + 15)) > 10, "Purple mesh missing"
    delta = np.abs(first - frames[-1].astype(np.int16)).mean()
    assert delta > 0.01, "Animation frames are identical"
    return {"path": str(path), "width": WIDTH, "height": HEIGHT, "frames": len(frames), "fps": fps, "motion_delta": float(delta)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=["world", "cam", "both"], default="both")
    args = parser.parse_args()
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="hawor-render-smoke-"))
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    background = np.full((HEIGHT, WIDTH, 3), 24, dtype=np.uint8)
    background[::24, :] = 40
    image_names = []
    for index in range(FRAMES):
        path = image_dir / f"{index:04d}.png"
        assert cv2.imwrite(str(path), background)
        image_names.append(str(path))
    rotations = np.repeat(np.eye(3, dtype=np.float32)[None], FRAMES, axis=0)
    translations = np.zeros((FRAMES, 3), dtype=np.float32)
    results = {}
    for mode in (["world", "cam"] if args.mode == "both" else [args.mode]):
        target = output_dir / mode
        export_dir = target / "aitviewer"
        export_dir.mkdir(parents=True, exist_ok=True)
        # Existing exports must be replaced at the exact returned path.
        (export_dir / "video.mp4").write_bytes(b"previous export")
        if mode == "world":
            left = synthetic_mesh((0.48, -0.68, -0.50), 0)
            right = synthetic_mesh((0.48, -0.32, -0.50), 0)
            path = run_vis2_on_video(left, right, str(target), 220, image_names,
                                     R_c2w=rotations, t_c2w=translations, interactive=False)
        else:
            left = synthetic_mesh((-0.28, 0, 2.0), 1)
            right = synthetic_mesh((0.28, 0, 2.0), 1)
            path = run_vis2_on_video_cam(left, right, str(target), 220, image_names,
                                         R_w2c=rotations, t_w2c=translations, interactive=False)
        assert Path(path) == export_dir / "video.mp4", f"Unexpected export path: {path}"
        results[mode] = check_video(path, expect_background=mode == "cam")
    (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
