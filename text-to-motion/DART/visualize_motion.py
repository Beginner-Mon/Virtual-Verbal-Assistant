import time
from pathlib import Path
import re

import numpy as np
import pyrender
import smplx
import torch
import trimesh


# ==========================
# CONFIG
# ==========================
INPUT_PATH = "outputs/motion_befe9e2a-850.glb"
MODEL_PATH = "./data/smplx_lockedhead_20230207/models_lockedhead"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLB_MULTI_GEOMETRY_MODE = "middle"  # middle | animate
GLB_PREVIEW_FPS = 30


def render_npz(npz_path: Path) -> None:
    """Render SMPL-X motion stored in NPZ (poses/trans/betas)."""
    data = np.load(str(npz_path), allow_pickle=True)
    print("Keys:", data.files)

    if "poses" not in data or "trans" not in data:
        raise ValueError("NPZ must contain 'poses' and 'trans' arrays")

    poses = data["poses"]
    trans = data["trans"]

    if "betas" in data:
        betas = data["betas"][:10]
    else:
        betas = np.zeros(10)

    print("Pose shape:", poses.shape)
    total_frames = poses.shape[0]

    model = smplx.create(
        MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
    ).to(DEVICE)

    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    scene = pyrender.Scene()
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

    mesh_node = None
    for frame_idx in range(total_frames):
        pose_t = torch.tensor(poses[frame_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        trans_t = torch.tensor(trans[frame_idx], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(
                betas=betas_t,
                global_orient=pose_t[:, :3],
                body_pose=pose_t[:, 3:66],
                left_hand_pose=pose_t[:, 66:111],
                right_hand_pose=pose_t[:, 111:156],
                jaw_pose=pose_t[:, 156:159],
                leye_pose=pose_t[:, 159:162],
                reye_pose=pose_t[:, 162:165],
                transl=trans_t,
            )

        vertices = output.vertices[0].cpu().numpy()
        mesh = trimesh.Trimesh(vertices, model.faces, process=False)
        render_mesh = pyrender.Mesh.from_trimesh(mesh)

        with viewer.render_lock:
            if mesh_node is not None:
                scene.remove_node(mesh_node)
            mesh_node = scene.add(render_mesh)

        time.sleep(1 / 30)

    print("NPZ animation finished.")


def render_glb(glb_path: Path) -> None:
    """Render a GLB/GLTF asset directly without NumPy loading."""
    loaded = trimesh.load(str(glb_path), force="scene")

    if isinstance(loaded, trimesh.Trimesh):
        tm_scene = trimesh.Scene(loaded)
    elif isinstance(loaded, trimesh.Scene):
        tm_scene = loaded
    else:
        raise ValueError(f"Unsupported GLB/GLTF content type: {type(loaded)}")

    if not tm_scene.geometry:
        raise ValueError("GLB/GLTF scene has no geometry to render")

    geometry_names = list(tm_scene.geometry.keys())

    def _frame_sort_key(name: str):
        match = re.search(r"frame_(\d+)", name)
        if match:
            return (0, int(match.group(1)))
        return (1, name)

    geometry_names.sort(key=_frame_sort_key)

    if len(geometry_names) > 1:
        print(
            f"Detected {len(geometry_names)} geometries in GLB. "
            f"Using multi-geometry mode: {GLB_MULTI_GEOMETRY_MODE}."
        )

        if GLB_MULTI_GEOMETRY_MODE == "animate":
            scene = pyrender.Scene()
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
            node = None
            frame_delay = 1.0 / max(1, GLB_PREVIEW_FPS)

            while viewer.is_active:
                for name in geometry_names:
                    mesh = tm_scene.geometry[name]
                    render_mesh = pyrender.Mesh.from_trimesh(mesh)
                    with viewer.render_lock:
                        if node is not None:
                            scene.remove_node(node)
                        node = scene.add(render_mesh)
                    if not viewer.is_active:
                        break
                    time.sleep(frame_delay)
            return

        middle_name = geometry_names[len(geometry_names) // 2]
        middle_mesh = tm_scene.geometry[middle_name]
        preview_scene = trimesh.Scene(middle_mesh)
        render_scene = pyrender.Scene.from_trimesh_scene(preview_scene)
        pyrender.Viewer(render_scene, use_raymond_lighting=True, run_in_thread=False)
        return

    render_scene = pyrender.Scene.from_trimesh_scene(tm_scene)
    pyrender.Viewer(render_scene, use_raymond_lighting=True, run_in_thread=False)


def main() -> None:
    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = input_path.suffix.lower()
    if ext == ".npz":
        render_npz(input_path)
    elif ext in {".glb", ".gltf"}:
        render_glb(input_path)
        print("GLB/GLTF render finished.")
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Use .npz, .glb, or .gltf")


if __name__ == "__main__":
    main()