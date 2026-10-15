"""Build synchronized comparisons from real HaWoR exports, without inference.

Inputs must already be 30 fps. The official demo drops the final input frame;
this tool preserves the exported frame range rather than duplicating a frame.
Optional tracks must be trusted model_tracks.npy files from your own run.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


def metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = dict(filename=path.name, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return info


def detector_crops(path, frames, width, height):
    boxes = [[] for _ in range(frames)]
    if path:
        tracks = np.load(path, allow_pickle=True).item()
        for records in tracks.values():
            for record in records:
                frame = int(record["frame"])
                if not record.get("det", False) or not 0 <= frame < frames:
                    continue
                for box in np.asarray(record["det_box"]).reshape(-1, 5):
                    if np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]:
                        boxes[frame].append(box[:4])
    roi = np.full((frames, 3), np.nan)
    detected = np.array([bool(b) for b in boxes])
    aspect = width / height
    for index, records in enumerate(boxes):
        if not records:
            continue
        b = np.asarray(records)
        x0, y0 = b[:, :2].min(axis=0)
        x1, y1 = b[:, 2:].max(axis=0)
        roi[index] = [(x0 + x1) / 2, (y0 + y1) / 2,
                      min(width, max((x1-x0)*1.35, (y1-y0)*aspect*1.35, width*.3))]
    if not detected.any():
        roi[:] = [width/2, height/2, width]
    else:
        present = np.where(detected)[0]
        for axis in range(3):
            roi[:, axis] = np.interp(np.arange(frames), present, roi[present, axis])
            roi[:, axis] = np.convolve(np.pad(roi[:, axis], (3, 3), mode="edge"), np.ones(7)/7, mode="valid")
    result = []
    for (cx, cy, crop_width), records in zip(roi, boxes):
        # Expand a smoothed crop if needed to retain the current detected boxes.
        if records:
            b = np.asarray(records)
            dx = np.abs(b[:, [0, 2]] - cx).max()
            dy = np.abs(b[:, [1, 3]] - cy).max()
            crop_width = max(crop_width, 2.2*dx, 2.2*dy*aspect)
        crop_width = min(width, crop_width)
        crop_height = crop_width/aspect
        cx = np.clip(cx, crop_width/2, width-crop_width/2)
        cy = np.clip(cy, crop_height/2, height-crop_height/2)
        result.append([int(cx-crop_width/2), int(cy-crop_height/2),
                       min(width, int(cx+crop_width/2)), min(height, int(cy+crop_height/2))])
    return result, detected


def canvas(panels, labels, title, footer, tile_width=480):
    tile_height = tile_width*9//16
    result = np.full((tile_height+70, tile_width*len(panels), 3), (25, 20, 16), np.uint8)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index*tile_width
        scale = min(tile_width/panel.shape[1], tile_height/panel.shape[0])
        size = (min(tile_width, round(panel.shape[1]*scale)), min(tile_height, round(panel.shape[0]*scale)))
        fitted = cv2.resize(panel, size, interpolation=cv2.INTER_AREA)
        dx, dy = (tile_width-size[0])//2, (tile_height-size[1])//2
        result[36+dy:36+dy+size[1], x+dx:x+dx+size[0]] = fitted
        cv2.putText(result, label, (x+12, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(result, f"{title} | {footer}", (12, tile_height+59),
                cv2.FONT_HERSHEY_SIMPLEX, .46, (215, 215, 215), 1, cv2.LINE_AA)
    return result


def encoder(path, width, height, fps, logfile):
    return subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "25", "-threads", "2",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=logfile,
    )


def verify(path, frames, width, height):
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        assert image.shape[:2] == (height, width), path
        count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert count == frames and abs(fps-30) < .01, (path, count, fps)
    return dict(metadata(path), decoded_frames=count, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "cam", "world", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--title", default="HaWoR")
    parser.add_argument("--credit", default="Input: ThunderVVV/HaWoR")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.input, args.cam, args.world]
    info = [metadata(p) for p in paths]
    n = info[1]["frames"]
    assert n > 0 and n == info[2]["frames"] and info[0]["frames"]-n in [0, 1], info
    assert all(abs(item["fps"]-30) < .01 for item in info), "Convert input to 30 fps before inference"
    assert all((item["width"], item["height"]) == (info[0]["width"], info[0]["height"]) for item in info)
    crops, detected = detector_crops(args.tracks, n, info[0]["width"], info[0]["height"])
    captures = [cv2.VideoCapture(str(p)) for p in paths]
    key_indices = set(np.linspace(0, n-1, 5).round().astype(int))
    stills = []
    with (args.output/"encoding.log").open("w") as log, ExitStack() as resources:
        for capture in captures:
            resources.callback(capture.release)
        wide = resources.enter_context(encoder(args.output/"comparison.mp4", 1440, 340, 30, log))
        detail = resources.enter_context(encoder(args.output/"hand_detail.mp4", 960, 340, 30, log))
        for frame in range(n):
            images = []
            for capture in captures:
                ok, image = capture.read()
                if not ok:
                    raise ValueError(f"Source decoding stopped at frame {frame}")
                images.append(image)
            footer = f"f{frame:03d}/{n-1:03d}  t={frame/30:.2f}s | L purple / R blue | {args.credit}"
            full = canvas(images, ["Input", "Camera overlay", "World: upstream fixed view"], args.title, footer)
            wide.stdin.write(full.tobytes())
            x0, y0, x1, y1 = crops[frame]
            roi_note = "detector crop" if detected[frame] else "crop interpolated (no raw detection)"
            close = canvas([im[y0:y1, x0:x1] for im in images[:2]],
                           ["Input: hand detail", "Camera overlay: hand detail"],
                           args.title, footer+" | "+roi_note)
            detail.stdin.write(close.tobytes())
            if frame in key_indices:
                stills.append(full)
                cv2.imwrite(str(args.output/f"detail_{frame:03d}.jpg"), close, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for capture, item in zip(captures, info):
            decoded = n
            while capture.read()[0]:
                decoded += 1
            assert decoded == item["frames"], (item["filename"], decoded, item["frames"])
            item["decoded_frames"] = decoded
    if wide.returncode or detail.returncode:
        raise RuntimeError(f"FFmpeg failed: see {args.output/'encoding.log'}")
    cv2.imwrite(str(args.output/"keyframes.jpg"), np.concatenate(stills), [cv2.IMWRITE_JPEG_QUALITY, 92])
    # Original exports are retained outside Git alongside compact comparisons.
    shutil.copyfile(args.cam, args.output/"camera_full.mp4")
    shutil.copyfile(args.world, args.output/"world_full.mp4")
    report = dict(title=args.title, sources=dict(zip(["input", "camera", "world"], info)),
                  alignment="Same 0-based frame indices; preserve upstream final-frame exclusion",
                  crop_note="Display-only detector-driven crop; never changes model predictions",
                  frames_with_raw_boxes=int(detected.sum()), crop_xyxy=crops,
                  outputs={"comparison": verify(args.output/"comparison.mp4", n, 1440, 340),
                           "hand_detail": verify(args.output/"hand_detail.mp4", n, 960, 340),
                           "camera_full": verify(args.output/"camera_full.mp4", n, info[1]["width"], info[1]["height"]),
                           "world_full": verify(args.output/"world_full.mp4", n, info[2]["width"], info[2]["height"])})
    (args.output/"media.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"title": args.title, "frames": n, "verified_outputs": 4, "output": str(args.output)}))


if __name__ == "__main__":
    main()
