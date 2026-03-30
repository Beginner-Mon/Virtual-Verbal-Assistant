import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import smplx
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SMPL-X NPZ motion to GLB.")
    parser.add_argument("--npz_path", default="outputs/motion_64e6c614-0c7.npz")
    parser.add_argument("--model_path", default="./data/smplx_lockedhead_20230207/models_lockedhead")
    parser.add_argument("--out_path", default="motion.glb")
    parser.add_argument("--gender", choices=["female", "male", "neutral"], default="female")
    parser.add_argument(
        "--mode",
        choices=["single", "stack", "sequence"],
        default="stack",
        help=(
            "single: export one representative frame to one GLB; "
            "stack: export many frame meshes in one GLB (default, frame_* nodes); "
            "sequence: export one GLB per frame."
        ),
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=-1,
        help="Frame index for single mode. Use -1 to auto-pick the middle frame.",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=1,
        help="Take every Nth frame for stack/sequence modes.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Cap number of exported frames in stack/sequence (0 means no cap).",
    )
    return parser.parse_args()


def _normalize_frame_index(frame_index: int, num_frames: int) -> int:
    if num_frames <= 0:
        raise ValueError("No frames found in input motion")
    if frame_index < 0:
        return num_frames // 2
    return max(0, min(frame_index, num_frames - 1))


def _frame_indices(num_frames: int, stride: int, max_frames: int) -> List[int]:
    step = max(1, int(stride))
    indices = list(range(0, num_frames, step))
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


def _build_frame_mesh(
    model,
    poses: np.ndarray,
    trans: np.ndarray,
    betas_t: torch.Tensor,
    frame_idx: int,
    device: torch.device,
) -> trimesh.Trimesh:
    pose = torch.tensor(poses[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)
    translation = torch.tensor(trans[frame_idx], dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        out = model(
            betas=betas_t,
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:66],
            left_hand_pose=pose[:, 66:111],
            right_hand_pose=pose[:, 111:156],
            jaw_pose=pose[:, 156:159],
            leye_pose=pose[:, 159:162],
            reye_pose=pose[:, 162:165],
            transl=translation,
        )

    vertices = out.vertices[0].cpu().numpy()
    faces = model.faces
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _export_scene_glb(scene: trimesh.Scene, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb_bytes = trimesh.exchange.gltf.export_glb(scene)
    with open(out_path, "wb") as file:
        file.write(glb_bytes)


def main():
    args = parse_args()
    device = torch.device("cpu")
    data = np.load(args.npz_path, allow_pickle=True)

    poses = data["poses"]
    trans = data["trans"]
    betas = data["betas"] if "betas" in data else np.zeros(10, dtype=np.float32)
    num_frames = poses.shape[0]

    model = smplx.create(
        args.model_path,
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
    ).to(device)

    betas_t = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)
    out_path = Path(args.out_path)

    if args.mode == "single":
        frame_idx = _normalize_frame_index(args.frame_index, num_frames)
        scene = trimesh.Scene()
        mesh = _build_frame_mesh(model, poses, trans, betas_t, frame_idx, device)
        scene.add_geometry(mesh, node_name=f"frame_{frame_idx}")
        _export_scene_glb(scene, out_path)
        print(f"Saved single-frame GLB: {out_path} (frame={frame_idx})")
        return

    indices = _frame_indices(num_frames, args.sample_stride, args.max_frames)
    if not indices:
        raise ValueError("No frames selected after applying sample_stride/max_frames")

    if args.mode == "stack":
        scene = trimesh.Scene()
        for frame_idx in indices:
            mesh = _build_frame_mesh(model, poses, trans, betas_t, frame_idx, device)
            scene.add_geometry(mesh, node_name=f"frame_{frame_idx}")
        _export_scene_glb(scene, out_path)
        print(f"Saved stacked GLB: {out_path} (frames={len(indices)})")
        return

    # sequence mode: one GLB per selected frame
    if out_path.suffix.lower() == ".glb":
        out_dir = out_path.parent
        stem = out_path.stem
    else:
        out_dir = out_path
        stem = Path(args.npz_path).stem

    out_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in indices:
        scene = trimesh.Scene()
        mesh = _build_frame_mesh(model, poses, trans, betas_t, frame_idx, device)
        scene.add_geometry(mesh, node_name=f"frame_{frame_idx}")
        frame_path = out_dir / f"{stem}_f{frame_idx:04d}.glb"
        _export_scene_glb(scene, frame_path)

    print(f"Saved GLB sequence: {out_dir} (frames={len(indices)})")


if __name__ == "__main__":
    main()
