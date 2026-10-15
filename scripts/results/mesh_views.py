#!/usr/bin/env python3
"""Render three fitted views of one saved HaWoR world-space MANO prediction.

Trusted local joblib/numpy caches only. CPU parametric MANO forward, no learned
inference. Exports a PNG and small metadata only, never mesh/pose/model arrays.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

COLORS = ("#a855f7", "#3b82f6")
SIDES = ("Left", "Right")
VIEWS = ((20, -60), (20, 60), (90, -90))


def load_cpu(path):
    original = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu")
    try:
        return joblib.load(path)
    finally:
        torch.storage._load_from_bytes = original


def choose_frame(valid, raw):
    """Nearest to (N-1)/2; deterministic tie selects earlier frame."""
    n = valid.shape[1]
    midpoint = (n - 1) / 2
    options = [(valid.all(0) & raw.all(0), "both raw detector sides present and both final-valid"),
               (valid.all(0), "fallback: both final-valid; both raw detector sides unavailable"),
               (valid.any(0), "fallback: at least one final-valid hand; invalid hand omitted")]
    for eligible, criterion in options:
        candidates = np.flatnonzero(eligible)
        if len(candidates):
            frame = int(candidates[np.argmin(np.abs(candidates - midpoint))])
            return frame, {"criterion": criterion, "midpoint_frame_index": midpoint,
                           "eligible_frame_count": len(candidates), "tie_break": "earlier frame"}
    raise ValueError("No final-valid frame is available")


def supports(sequence, n):
    folders = sorted(sequence.glob("tracks_*"))
    if len(folders) != 1:
        raise ValueError(f"Expected one tracks_* folder, found {len(folders)}")
    folder = folders[0]
    raw = np.zeros((2, n), bool)
    records = np.load(folder / "model_tracks.npy", allow_pickle=True).item()
    for track in records.values():
        for record in track:
            t = int(record["frame"])
            if record["det"] and 0 <= t < n:
                raw[int(np.asarray(record["det_handedness"]).ravel()[0] > 0), t] = True
    camera = np.zeros((2, n), bool)
    chunks_path = folder / "frame_chunks_all.npy"
    if not chunks_path.exists():
        return raw, None
    chunks = load_cpu(chunks_path)
    for side in range(2):
        for frames in chunks.get(side, []):
            frames = np.asarray(frames, int)
            path = sequence / "cam_space" / str(side) / f"{frames[0]}_{frames[-1]}.json"
            if path.exists():
                camera[side, frames] = True
    return raw, camera


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="HaWoR checkout with installed MANO assets")
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Diagnostics directory outside source cache")
    parser.add_argument("--attribution", default="", help="Input credit shown verbatim in figure footer")
    args = parser.parse_args()
    repo, sequence, output = args.repo.resolve(), args.sequence.resolve(), args.output.resolve()
    if output == sequence or sequence in output.parents:
        parser.error("--output must be outside source cache")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from hawor.utils.process import run_mano, run_mano_left
    values = load_cpu(sequence / "world_space_res.pth")
    if len(values) != 5:
        raise ValueError("Expected joblib [trans,rot,hand_pose,betas,valid]")
    tensors = [torch.as_tensor(value).detach().cpu() for value in values[:4]]
    valid_values = np.asarray(values[4])
    valid = np.isfinite(valid_values) & (valid_values > 0)
    if valid.ndim != 2 or valid.shape[0] != 2:
        raise ValueError(f"Invalid validity shape {valid.shape}")
    n = valid.shape[1]
    raw, camera = supports(sequence, n)
    frame, selection = choose_frame(valid, raw)
    meshes, faces, details = [], [], {}
    with torch.no_grad():
        for side, forward in enumerate((run_mano_left, run_mano)):
            if not valid[side, frame]:
                details[SIDES[side].lower()] = {"rendered": False, "reason": "final-invalid"}
                continue
            trans, rot, pose, betas = [x[side:side+1, frame:frame+1].float() for x in tensors]
            if not all(torch.isfinite(x).all() for x in (trans, rot, pose, betas)):
                raise ValueError(f"Nonfinite saved parameters for side {side} frame {frame}")
            result = forward(trans, rot, pose, is_right=torch.full((1,1,1), float(side)), betas=betas, use_cuda=False)
            vertices = result["vertices"][0, 0].detach().cpu().numpy()
            triangles = result["faces"][0, 0].detach().cpu().numpy().astype(int)
            if not np.isfinite(vertices).all() or triangles.min() < 0 or triangles.max() >= len(vertices):
                raise ValueError("Invalid generated mesh geometry")
            meshes.append((side, vertices))
            faces.append(triangles)
            details[SIDES[side].lower()] = {"rendered": True, "vertex_count": len(vertices),
                "face_count": len(triangles), "finite_vertices": bool(np.isfinite(vertices).all()),
                "vertex_axis_min": vertices.min(0).tolist(), "vertex_axis_max": vertices.max(0).tolist(),
                "raw_detector_label_present": bool(raw[side, frame]), "final_valid": bool(valid[side, frame]),
                "camera_stage_support": bool(camera[side, frame]) if camera is not None else None,
                "forward_function": forward.__name__, "color": COLORS[side]}
    xyz = np.concatenate([vertices for side, vertices in meshes])
    low, high = xyz.min(0).astype(float), xyz.max(0).astype(float)
    center = (low + high) / 2
    half_width = max(float((high - low).max()) * .56, 1e-6)
    limits = np.stack([center-half_width, center+half_width], axis=1)
    polygons = np.concatenate([vertices[triangles] for (side, vertices), triangles in zip(meshes, faces)])
    face_colors = [COLORS[side] for (side, vertices), triangles in zip(meshes, faces) for _ in triangles]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelsize": 10})
    fig = plt.figure(figsize=(16.2, 6.4))
    for column, (elev, azim) in enumerate(VIEWS, 1):
        ax = fig.add_subplot(1, 3, column, projection="3d")
        surface = Poly3DCollection(polygons, facecolors=face_colors, edgecolors="#252d402e", linewidths=.13, alpha=1.0, zsort="average")
        ax.add_collection3d(surface)
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
        ax.set_box_aspect((1,1,1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("World X", labelpad=6)
        ax.set_ylabel("World Y", labelpad=6)
        ax.set_zlabel("World Z", labelpad=6)
        if elev == 90:
            # Looking along Z collapses all Z ticks onto one screen location.
            # Hide those ticks without changing the shared 3D limits/geometry.
            ax.set_zticks([])
            ax.set_zlabel("")
        ax.tick_params(labelsize=8, pad=0)
        ax.set_title(f"View {column} | elevation {elev} deg, azimuth {azim} deg", fontsize=11, pad=8)
    fig.suptitle(f"{sequence.name} | final predicted MANO meshes | extracted frame {frame} / {n-1}", x=.025, ha="left", y=.98, fontsize=17)
    fig.legend(handles=[Patch(facecolor=COLORS[i], label=SIDES[i]) for i, _ in meshes], loc="upper right", bbox_to_anchor=(.98,.99), frameon=False, ncol=2, fontsize=11)
    footer = "Stored world coordinates / stored units. Same fitted equal-axis limits in all views; mesh geometry unchanged. No ground truth.\nNearest eligible midpoint frame: both raw detector sides + both final-valid preferred. Final predictions may include infill."
    if args.attribution:
        footer += "\n" + args.attribution
    fig.text(.025, .035, footer, fontsize=10, color="#475569", va="bottom")
    fig.subplots_adjust(left=.015, right=.99, top=.86, bottom=.19, wspace=.04)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "04_mesh_views.png", dpi=200, facecolor="white")
    plt.close(fig)
    metadata = {"schema_version": 1, "sequence": str(sequence), "frame_index": frame,
        "frame_count": n, "frame_index_definition": "0-based extracted-image index, not source video timestamp",
        "frame_selection": selection, "hands": details,
        "geometry_definition": "Actual run_mano_left/run_mano CPU forward of saved final world trans/angle-axis root/hand pose/betas; same left shapedirs fix and wrist-cap faces as repository helper defaults",
        "coordinate_definition": "Original saved-world MANO vertices from the parametric forward; no demo rotation, recentering, scaling, smoothing or pose changes. Only plot limits/camera view are fitted.",
        "units": "stored world units; physical scale not independently validated",
        "axis_order": ["X", "Y", "Z"], "common_axis_limits": limits.tolist(),
        "axis_fit_definition": "Union vertex bounding-box center; all 3 axes half-width 0.56 * largest bounding-box side; equal-axis cube shared by 3 views",
        "views": [{"elevation_degrees": e, "azimuth_degrees": a} for e,a in VIEWS],
        "projection": "orthographic", "rendering": "Matplotlib Poly3DCollection, opaque constant side color, thin triangle edges, no shading or lighting; painter-based static display",
        "limitations": ["No ground truth, no measured reconstruction accuracy or anatomical plausibility evaluation.",
                        "Raw-side detector support does not prove hand identity, visibility, or absence of infill after track-majority reassignment.",
                        "Parametric MANO forward only; this does not rerun learned reconstruction or modify saved inference outputs.",
                        "Matplotlib polygon depth ordering can have occlusion artifacts; geometry is not changed to improve appearance."],
        "attribution": args.attribution or None, "cuda_initialized": torch.cuda.is_initialized(),
        "provenance": {"world_cache": str(sequence/"world_space_res.pth"), "world_cache_sha256": digest(sequence/"world_space_res.pth"),
                       "script_sha256": digest(Path(__file__).resolve()), "process_py_sha256": digest(repo/"hawor/utils/process.py"),
                       "torch": torch.__version__, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "exported": ["04_mesh_views.png", "mesh_view.json"],
        "not_exported": "Mesh vertices, triangle arrays, saved pose/shape parameters, MANO assets and learned weights"}
    (output/"mesh_view.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"case": sequence.name, "frame": frame, "output": str(output/"04_mesh_views.png"), "cuda_initialized": torch.cuda.is_initialized()}))


if __name__ == "__main__":
    main()
