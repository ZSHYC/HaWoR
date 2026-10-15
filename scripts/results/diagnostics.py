#!/usr/bin/env python3
"""Offline CPU diagnostics for trusted HaWoR video caches; no accuracy metrics.

Only load caches produced by a trusted inference run: joblib/numpy pickle files
can execute code. Does not import HaWoR, load MANO/model weights or run inference.
"""

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image
import torch

NAMES = ("pred_trans", "pred_rot", "pred_hand_pose", "pred_betas", "pred_valid")
SIDES = ("Left", "Right")
COLORS = ("#a855f7", "#3b82f6")


@contextmanager
def cpu_storage_load():
    """Map tensor storages in trusted joblib pickle files to CPU."""
    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
    try:
        yield
    finally:
        torch.storage._load_from_bytes = original


def load_joblib_cpu(path):
    with cpu_storage_load():
        return joblib.load(path)


def as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def true_ranges(mask):
    """Inclusive [first,last] frame ranges of True runs."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return [[int(a), int(b - 1)] for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


def adjacent_displacement(x, valid):
    """At frame t>0: ||x[t]-x[t-1]||2 iff both adjacent frames valid/finite."""
    x = np.asarray(x, dtype=np.float64)
    good = np.asarray(valid, bool) & np.isfinite(x).all(axis=-1)
    result = np.full(len(x), np.nan)
    pairs = good[1:] & good[:-1]
    result[1:] = np.where(pairs, np.linalg.norm(np.diff(x, axis=0), axis=-1), np.nan)
    return result


def scalar_stats(value):
    a = np.asarray(value, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {"count": int(a.size), "min": float(a.min()), "median": float(np.median(a)),
            "mean": float(a.mean()), "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def array_stats(value):
    a = as_numpy(value)
    finite = np.isfinite(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "elements": int(a.size),
            "finite": int(finite.sum()), "nonfinite": int((~finite).sum()),
            "finite_fraction": float(finite.mean()) if a.size else None,
            "finite_value_summary": scalar_stats(a)}


def coverage(mask):
    mask = np.asarray(mask, bool)
    return {"frames": int(mask.size), "present_frames": int(mask.sum()),
            "fraction": float(mask.mean()) if mask.size else None,
            "present_ranges_inclusive": true_ranges(mask),
            "absent_ranges_inclusive": true_ranges(~mask)}


def fingerprint(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def unique_path(paths, what, explicit=None):
    paths = sorted(paths)
    if explicit:
        return Path(explicit).resolve()
    if len(paths) > 1:
        raise ValueError(f"Multiple {what} candidates: {paths}; provide an explicit path")
    return paths[0] if paths else None


def load_tracks(folder, n, image_size, warnings):
    if folder is None or not (folder / "model_tracks.npy").exists():
        warnings.append("Raw detection tracks unavailable")
        return None, None, None, {}, []
    path = folder / "model_tracks.npy"
    data = np.load(path, allow_pickle=True).item()
    label_support = np.zeros((2, n), bool)
    majority_support = np.zeros((2, n), bool)
    confidence = np.full((2, n), np.nan)
    summaries, all_boxes, invalid_frames = [], [], []
    for key, entries in data.items():
        detected = [e for e in entries if bool(e.get("det", False))]
        labels = np.concatenate([np.asarray(e["det_handedness"]).reshape(-1) for e in detected]) if detected else np.array([])
        majority = int(labels.mean() >= .5) if labels.size else None
        frames = [int(e["frame"]) for e in entries]
        disagreements = 0
        for e in detected:
            frame = int(e["frame"])
            label = int(np.asarray(e["det_handedness"]).ravel()[0] > 0)
            boxes = np.asarray(e["det_box"], float).reshape(-1, 5)
            all_boxes.extend(boxes.tolist())
            disagreements += int(majority != label)
            if not 0 <= frame < n:
                invalid_frames.append(frame)
                continue
            label_support[label, frame] = True
            majority_support[majority, frame] = True
            score = float(boxes[0, 4])
            if np.isfinite(score):
                confidence[label, frame] = max(score, confidence[label, frame]) if np.isfinite(confidence[label, frame]) else score
        summaries.append({"track_id": str(key), "entries": len(entries), "detected_entries": len(detected),
                          "first_frame": min(frames) if frames else None, "last_frame": max(frames) if frames else None,
                          "duplicate_frame_entries": len(frames) - len(set(frames)),
                          "majority_assigned_side": SIDES[majority].lower() if majority is not None else None,
                          "raw_label_left_count": int((labels == 0).sum()),
                          "raw_label_right_count": int((labels == 1).sum()),
                          "raw_label_vs_track_majority_disagreements": disagreements})
    boxes = np.asarray(all_boxes, float).reshape(-1, 5)
    box_report = {"detected_entries": len(boxes), "schema": "[x1,y1,x2,y2,detector_score] in source image pixels",
                  "array": array_stats(boxes), "score": scalar_stats(boxes[:, 4]),
                  "negative_or_zero_width_or_height": int(((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).sum()),
                  "score_outside_0_1": int(((boxes[:, 4] < 0) | (boxes[:, 4] > 1)).sum())}
    if image_size:
        width, height = image_size
        box_report["boxes_outside_image_extent"] = int(((boxes[:, 0] < 0) | (boxes[:, 1] < 0) | (boxes[:, 2] > width) | (boxes[:, 3] > height)).sum())
    model_boxes = folder / "model_boxes.npy"
    if model_boxes.exists():
        cached_boxes = np.load(model_boxes, allow_pickle=True)
        box_report["model_boxes_npy_shape"] = list(cached_boxes.shape)
        if cached_boxes.size == 0:
            warnings.append("model_boxes.npy is empty; raw detections reconstructed from model_tracks.npy instead")
    if invalid_frames:
        warnings.append(f"Track entries outside prediction frame domain: {invalid_frames}")
    report = {"track_summaries": summaries, "boxes": box_report,
              "raw_label_support": {side.lower(): coverage(label_support[i]) for i, side in enumerate(SIDES)},
              "track_majority_support": {side.lower(): coverage(majority_support[i]) for i, side in enumerate(SIDES)},
              "any_raw_detection": coverage(label_support.any(axis=0)),
              "both_raw_labels_present": coverage(label_support.all(axis=0)),
              "out_of_range_frame_entries": invalid_frames}
    return label_support, majority_support, confidence, report, [path]


def load_camera(sequence, tracks_dir, n, warnings):
    translations = np.full((2, n, 3), np.nan)
    support = np.zeros((2, n), bool)
    chunks = None
    files, info = [], []
    chunk_path = tracks_dir / "frame_chunks_all.npy" if tracks_dir else None
    if chunk_path and chunk_path.exists():
        chunks = load_joblib_cpu(chunk_path)
        files.append(chunk_path)
    for i in range(2):
        for path in sorted((sequence / "cam_space" / str(i)).glob("*.json")):
            start, end = [int(x) for x in path.stem.split("_")]
            frames = np.arange(start, end + 1)
            if chunks is not None:
                matched = [np.asarray(c, dtype=int) for c in chunks.get(i, []) if int(c[0]) == start and int(c[-1]) == end]
                if len(matched) != 1:
                    warnings.append(f"Ignoring camera JSON not uniquely matched in current frame_chunks_all: {path}")
                    continue
                frames = matched[0]
            data = {key: np.asarray(value) for key, value in json.loads(path.read_text()).items()}
            trans = data["init_trans"][0]
            if trans.shape != (len(frames), 3) or np.any(frames < 0) or np.any(frames >= n):
                raise ValueError(f"Camera JSON frame/translation shape mismatch: {path}")
            if support[i, frames].any():
                warnings.append(f"Overlapping camera JSON frame support: {path}")
            translations[i, frames] = trans
            support[i, frames] = True
            info.append({"path": str(path), "hand": SIDES[i].lower(), "frames": frames.tolist(),
                         "arrays": {key: array_stats(value) for key, value in data.items()}})
            files.append(path)
    if not info:
        warnings.append("Camera-space JSON estimates unavailable")
        return translations, None, {"chunks": []}, files
    return translations, support, {"chunks": info, "coverage": {side.lower(): coverage(support[i]) for i, side in enumerate(SIDES)}}, files


def load_masks(folder, n, warnings):
    path = folder / "model_masks.npy" if folder else None
    if path is None or not path.exists():
        return None, {"available": False}
    masks = np.load(path, mmap_mode="r", allow_pickle=False)
    if masks.ndim != 3 or len(masks) != n:
        warnings.append(f"Mask shape {masks.shape} does not match N={n}; skipping mask area")
        return None, {"available": True, "shape": list(masks.shape), "compatible": False}
    fraction = np.array([np.count_nonzero(m) / m.size for m in masks])
    return fraction, {"available": True, "path": str(path), "shape": list(masks.shape), "dtype": str(masks.dtype),
                      "definition": "nonzero rendered union-mask pixels / (image height * image width), per frame",
                      "area_fraction": scalar_stats(fraction)}


def load_slam(path, n, warnings):
    if path is None:
        return None, {"available": False}, []
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    report = {"available": True, "path": str(path), "arrays": {k: array_stats(v) for k, v in data.items()},
              "coordinate_definition": "traj[:, :3] * scale (same camera-to-world translation used by load_slam_cam)",
              "units": "stored units; physical scale not independently validated"}
    traj = data["traj"]
    if traj.shape != (n, 7) or np.asarray(data["scale"]).size != 1:
        warnings.append("SLAM trajectory shape/scale incompatible; skipping camera path")
        return None, report, [path]
    xyz = traj[:, :3].astype(float) * float(data["scale"])
    report["quaternion_norm"] = scalar_stats(np.linalg.norm(traj[:, 3:], axis=1))
    report["scaled_camera_translation"] = array_stats(xyz)
    report["adjacent_displacement"] = scalar_stats(adjacent_displacement(xyz, np.ones(n, bool)))
    return xyz, report, [path]


def motion_stats(trans, valid):
    report = {}
    for i, side in enumerate(SIDES):
        good = valid[i] & np.isfinite(trans[i]).all(axis=-1)
        step = adjacent_displacement(trans[i], good)
        xyz = trans[i, good]
        report[side.lower()] = {"usable_translation_frames": int(good.sum()),
            "axis_min": xyz.min(axis=0).tolist() if len(xyz) else None,
            "axis_max": xyz.max(axis=0).tolist() if len(xyz) else None,
            "axis_range": np.ptp(xyz, axis=0).tolist() if len(xyz) else None,
            "adjacent_displacement": scalar_stats(step),
            "sum_adjacent_displacement": float(np.nansum(step)),
            "zero_translation_valid_frames": np.where(good & (trans[i] == 0).all(axis=-1))[0].tolist()}
    return report


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "figure.titlesize": 16, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})


def save(fig, output, name):
    fig.savefig(output / name, facecolor="white")
    plt.close(fig)


def plot_support(output, name, n, raw, majority, camera, valid, confidence, area):
    rows, labels = [], []
    for title, support in [("Raw detector label", raw), ("Track-majority side", majority),
                           ("Camera-stage cache", camera), ("Final output valid", valid)]:
        for i, side in enumerate(SIDES):
            rows.append(support[i].astype(float) if support is not None else np.full(n, np.nan))
            count = f"{int(support[i].sum())}/{n}" if support is not None else "unavailable"
            labels.append(f"{title} | {side}  ({count})")
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.0), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.2]})
    cmap = ListedColormap(["#e7eaf0", "#2a837b"])
    cmap.set_bad("#b8b8b8")
    axes[0].imshow(np.ma.masked_invalid(np.array(rows)), aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=1, extent=(-.5, n-.5, len(rows)-.5, -.5))
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_title("Frame support: teal = present, pale = absent, gray = unavailable", loc="left")
    axes[0].set_yticks(np.arange(-.5, len(rows), 2), minor=True)
    axes[0].grid(which="minor", axis="y", color="white", linewidth=3)
    for i, side in enumerate(SIDES):
        if confidence is not None:
            axes[1].plot(range(n), confidence[i], color=COLORS[i], label=side, lw=1.6)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Detector score")
    axes[1].set_title("Maximum cached detector score per raw side label / frame", loc="left")
    if confidence is not None:
        axes[1].legend(loc="lower left", ncol=2, frameon=False)
    if area is not None:
        axes[2].plot(range(n), area * 100, color="#595870", lw=1.7)
        axes[2].fill_between(range(n), area * 100, color="#595870", alpha=.12)
    axes[2].set_ylabel("Image area (%)")
    axes[2].set_title("Rendered hand union-mask area (model-derived)", loc="left")
    axes[2].set_xlabel("Extracted frame index (0-based)")
    axes[2].set_xlim(-.5, n-.5)
    for ax in axes[1:]:
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(f"{name} | detection, cache support and final validity", x=.015, ha="left", y=.98)
    fig.text(.015, .025, "Final valid includes infill; camera-stage support can include interpolated boxes. Scores are not calibrated confidence.\nNo ground truth: support and mask area do not measure reconstruction accuracy.", fontsize=9, color="#4b4d54")
    fig.subplots_adjust(left=.29, right=.98, top=.89, bottom=.15, hspace=.55)
    save(fig, output, "01_support_and_detection.png")


def plot_translation(output, name, trans, camera_trans, valid):
    n = trans.shape[1]
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    for axis in range(3):
        for i, side in enumerate(SIDES):
            world = np.where(valid[i], trans[i, :, axis], np.nan)
            axes[axis, 0].plot(range(n), world, color=COLORS[i], label=side, lw=1.7)
            axes[axis, 1].plot(range(n), camera_trans[i, :, axis], color=COLORS[i], label=side, lw=1.7)
        for ax in axes[axis]:
            ax.grid(alpha=.2)
            ax.set_ylabel(f"{'XYZ'[axis]} translation\n(stored units)")
    axes[0, 0].set_title("World output | includes infill")
    axes[0, 1].set_title("Camera-stage estimates | gaps preserved")
    for ax in axes[-1]:
        ax.set_xlabel("Extracted frame index (0-based)")
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle(f"{name} | hand model translation components", x=.015, ha="left", y=.98)
    fig.text(.015, .025, "Translation parameter trajectories; not reconstructed wrist/joint positions. World and camera axes use different coordinate systems.\nFinal-invalid world frames are omitted. Physical units and scale are not independently validated; no ground-truth accuracy claim.", fontsize=9, color="#4b4d54")
    fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.14, hspace=.25, wspace=.3)
    save(fig, output, "02_translation_components.png")


def plot_motion(output, name, trans, valid, camera, slam):
    n = trans.shape[1]
    fig = plt.figure(figsize=(12, 6.4))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax = fig.add_subplot(1, 2, 2)
    points = []
    for i, side in enumerate(SIDES):
        good = valid[i] & np.isfinite(trans[i]).all(axis=-1)
        xyz = np.where(good[:, None], trans[i], np.nan)
        ax3.plot(*xyz.T, color=COLORS[i], label=side, lw=1.7)
        idx = np.where(good)[0]
        if idx.size:
            ax3.scatter(*xyz[idx[0]], color=COLORS[i], marker="o", s=30)
            ax3.scatter(*xyz[idx[-1]], color=COLORS[i], marker="^", s=45)
            points.append(xyz[good])
        steps = adjacent_displacement(trans[i], good)
        ax.plot(range(n), steps, color=COLORS[i], lw=1.6, label=side)
        if camera is not None:
            touched = (~camera[i]) & good & np.isfinite(steps)
            touched[1:] |= (~camera[i, :-1]) & good[1:] & np.isfinite(steps[1:])
            ax.scatter(np.where(touched)[0], steps[touched], color=COLORS[i], facecolors="white", s=22, zorder=4)
    if slam is not None:
        ax3.plot(*slam.T, color="#6b6c75", ls="--", lw=1.3, label="SLAM camera x scale")
        points.append(slam[np.isfinite(slam).all(axis=-1)])
    if points and any(len(p) for p in points):
        all_points = np.concatenate(points)
        mins, maxs = all_points.min(axis=0), all_points.max(axis=0)
        center = (mins + maxs) / 2
        radius = max(np.ptp(all_points, axis=0).max() * .55, .01)
        ax3.set_xlim(center[0]-radius, center[0]+radius)
        ax3.set_ylim(center[1]-radius, center[1]+radius)
        ax3.set_zlim(center[2]-radius, center[2]+radius)
        ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel("World X (stored units)", labelpad=8)
    ax3.set_ylabel("World Y (stored units)", labelpad=8)
    ax3.set_zlabel("World Z (stored units)", labelpad=8)
    ax3.view_init(elev=22, azim=-65)
    ax3.legend(loc="upper left", frameon=False, fontsize=9)
    ax3.set_title("Final-valid hand model translations\nCircle = first; triangle = last valid frame", fontsize=11)
    ax.set_title("Consecutive-frame translation displacement", fontsize=11)
    ax.set_xlabel("Extracted frame index (0-based)")
    ax.set_ylabel("Euclidean displacement (stored units / frame)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.suptitle(f"{name} | world trajectory and framewise motion", x=.015, ha="left", y=.98)
    fig.text(.015, .025, "Displacement uses only adjacent final-valid finite predictions; open markers touch a frame without camera-stage support.\nNot velocity, jitter, or accuracy: motion and infill both contribute. 3D view has equal axis scale; physical scale is unverified.", fontsize=9, color="#4b4d54")
    fig.subplots_adjust(left=.04, right=.98, top=.84, bottom=.2, wspace=.25)
    save(fig, output, "03_world_trajectory_and_motion.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path, help="HaWoR case cache directory")
    parser.add_argument("--output", required=True, type=Path, help="New diagnostics destination outside caches")
    parser.add_argument("--tracks-dir", type=Path, help="Explicit tracks_N_M directory if multiple exist")
    parser.add_argument("--slam", type=Path, help="Explicit SLAM NPZ if multiple exist")
    args = parser.parse_args()
    sequence = args.sequence.resolve()
    output = args.output.resolve()
    if output == sequence or sequence in output.parents:
        parser.error("--output must be outside the source cache directory")
    output.mkdir(parents=True, exist_ok=True)
    warnings = []
    world_path = sequence / "world_space_res.pth"
    values = load_joblib_cpu(world_path)
    if not isinstance(values, (list, tuple)) or len(values) != 5:
        raise ValueError("Expected HaWoR joblib list [trans,rot,hand_pose,betas,valid]")
    arrays = {name: as_numpy(value) for name, value in zip(NAMES, values)}
    trans = arrays["pred_trans"]
    if trans.ndim != 3 or trans.shape[0] != 2 or trans.shape[2] != 3:
        raise ValueError(f"Unexpected translation shape: {trans.shape}")
    n = trans.shape[1]
    expected = {"pred_trans": (2,n,3), "pred_rot": (2,n,3), "pred_hand_pose": (2,n,45), "pred_betas": (2,n,10), "pred_valid": (2,n)}
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"Unexpected {key} shape {arrays[key].shape}; expected {shape}")
        if not np.isfinite(arrays[key]).all():
            warnings.append(f"Nonfinite values found in {key}; see array statistics")
    valid_values = arrays["pred_valid"]
    valid = np.isfinite(valid_values) & (valid_values > 0)
    if not np.isin(valid_values, [0,1]).all():
        warnings.append("pred_valid contains values outside {0,1}; finite positive values treated as valid")
    images = sorted((sequence / "extracted_images").glob("*.jpg"))
    image_size = None
    if images:
        with Image.open(images[0]) as image:
            image_size = image.size
        if len(images) != n:
            warnings.append(f"Extracted JPG count {len(images)} != prediction length {n}")
    tracks = unique_path(sequence.glob("tracks_*"), "tracks directories", args.tracks_dir)
    raw, majority, confidence, track_report, track_files = load_tracks(tracks, n, image_size, warnings)
    cam_trans, cam_support, cam_report, cam_files = load_camera(sequence, tracks, n, warnings)
    area, mask_report = load_masks(tracks, n, warnings)
    slam_path = unique_path((sequence / "SLAM").glob("hawor_slam_w_scale_*.npz"), "SLAM files", args.slam)
    slam, slam_report, slam_files = load_slam(slam_path, n, warnings)
    framewise = {"frame_index": list(range(n)), "final_valid": valid.tolist(),
                 "raw_label_support": raw.tolist() if raw is not None else None,
                 "track_majority_support": majority.tolist() if majority is not None else None,
                 "camera_stage_support": cam_support.tolist() if cam_support is not None else None,
                 "raw_label_max_detector_score": [[float(v) if np.isfinite(v) else None for v in row] for row in confidence] if confidence is not None else None,
                 "rendered_union_mask_area_fraction": area.tolist() if area is not None else None,
                 "world_adjacent_displacement": [[float(v) if np.isfinite(v) else None for v in adjacent_displacement(trans[i], valid[i])] for i in range(2)]}
    stage = {}
    if cam_support is not None:
        for i, side in enumerate(SIDES):
            stage[side.lower()] = {"final_valid_without_camera_stage": coverage(valid[i] & ~cam_support[i]),
                                  "camera_stage_but_final_invalid": coverage(cam_support[i] & ~valid[i])}
    report = {"schema_version": 1, "sequence": str(sequence), "frame_count": n,
              "side_order": [s.lower() for s in SIDES], "frame_indexing": "0-based indices of extracted images; not source video timestamps",
              "extracted_jpg_count": len(images), "first_image_size_width_height": list(image_size) if image_size else None,
              "arrays": {key: array_stats(value) for key, value in arrays.items()},
              "final_validity": {s.lower(): coverage(valid[i]) for i, s in enumerate(SIDES)},
              "detections": track_report, "camera_stage": cam_report, "stage_comparison": stage,
              "model_masks": mask_report, "slam": slam_report,
              "world_translation": motion_stats(trans, valid),
              "camera_translation": motion_stats(cam_trans, np.isfinite(cam_trans).all(axis=-1)),
              "definitions": {
                  "validity": "Final pred_valid from HaWoR; can be set True by infiller, not detector reliability or correctness",
                  "raw_label_support": "At least one model_tracks entry with det=True and raw det_handedness >0 (right) or ==0 (left), per frame",
                  "track_majority_support": "Assign each track left if mean of its det=True handedness labels <0.5, else right; union det=True frames by assigned side (same rule as HaWoR)",
                  "camera_stage_support": "Frames present in camera-space JSON matched to current frame_chunks_all when available; can include interpolated detection boxes",
                  "final_valid_without_camera_stage": "Final-valid frames absent from camera-stage cache; consistent with infill availability, not ground-truth visibility",
                  "translation": "Stored MANO model translation parameter, not wrist/joint positions; world uses cam2world_convert offset convention",
                  "adjacent_displacement": "L2 norm(trans[t]-trans[t-1]) only if both adjacent frames are final-valid and have finite xyz; no bridging gaps",
                  "sum_adjacent_displacement": "Sum of eligible consecutive-frame displacements; does not bridge gaps; stored units",
                  "axis_bounds": "Min/max/range over usable final-valid finite translation frames; invalid zero placeholders excluded",
                  "array_bounds": "Array numeric summaries include placeholders; finite value min/max are diagnostic, not anatomical limits",
                  "detector_score": "Cached detector score; maximum for a raw side label within a frame; not calibrated confidence",
                  "percentile": "p95 computed by numpy.quantile with default linear interpolation",
              },
              "limitations": ["No ground truth or benchmark annotations; no reconstruction accuracy, MPJPE, PVE, reprojection error, identity accuracy, or visibility accuracy is measured.",
                              "Metrics describe one saved prediction; finite tensors and full validity do not establish correctness.",
                              "Translation axes are stored world/camera coordinates. Units, world orientation and physical scale are not independently validated.",
                              "Consecutive-frame displacement is motion magnitude, not velocity or jitter; frame time is not assumed.",
                              "Rendered model-mask area is model-derived and is not segmentation IoU.",
                              "Raw tracks contain only retained detector/tracker records; they are not every pre-filter detector proposal.",
                              "Model weights, MANO assets and raw meshes are not loaded or exported."],
              "warnings": warnings,
              "provenance": {"world_serialization": "joblib list despite .pth suffix", "inputs": [fingerprint(p) for p in [world_path] + track_files + cam_files + slam_files],
                             "script": fingerprint(Path(__file__).resolve()),
                             "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}},
              "artifacts": ["01_support_and_detection.png", "02_translation_components.png", "03_world_trajectory_and_motion.png", "summary.json", "framewise.json"]}
    for name, content in [("summary.json", report), ("framewise.json", framewise)]:
        (output / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    style()
    plot_support(output, sequence.name, n, raw, majority, cam_support, valid, confidence, area)
    plot_translation(output, sequence.name, trans, cam_trans, valid)
    plot_motion(output, sequence.name, trans, valid, cam_support, slam)
    print(json.dumps({"output": str(output), "frames": n, "final_valid": valid.sum(axis=1).tolist(),
                      "raw_label_support": raw.sum(axis=1).tolist() if raw is not None else None,
                      "camera_stage_support": cam_support.sum(axis=1).tolist() if cam_support is not None else None,
                      "warnings": warnings}))


if __name__ == "__main__":
    main()
