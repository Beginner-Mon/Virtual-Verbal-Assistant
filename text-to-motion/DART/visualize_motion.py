import numpy as np
import torch
import smplx
import pyrender
import trimesh
import time

# ==========================
# CONFIG
# ==========================
NPZ_PATH = "outputs/motion_cb642a90-562.npz"
MODEL_PATH = "./data/smplx_lockedhead_20230207/models_lockedhead"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# LOAD MOTION
# ==========================
data = np.load(NPZ_PATH, allow_pickle=True)
print("Keys:", data.files)

poses = data["poses"]      # expect (T, 165)
trans = data["trans"]

if "betas" in data:
    betas = data["betas"][:10]
else:
    betas = np.zeros(10)

print("Pose shape:", poses.shape)

T = poses.shape[0]

# ==========================
# LOAD SMPL-X MODEL
# ==========================
model = smplx.create(
    MODEL_PATH,
    model_type="smplx",
    gender="neutral",
    use_pca=False
).to(DEVICE)

betas = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# ==========================
# RENDER SETUP
# ==========================
scene = pyrender.Scene()
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)

mesh_node = None

# ==========================
# ANIMATION
# ==========================
for t in range(T):

    pose = torch.tensor(poses[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    translation = torch.tensor(trans[t], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(
            betas=betas,
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:66],
            left_hand_pose=pose[:, 66:111],
            right_hand_pose=pose[:, 111:156],
            jaw_pose=pose[:, 156:159],
            leye_pose=pose[:, 159:162],
            reye_pose=pose[:, 162:165],
            transl=translation
        )

    vertices = output.vertices[0].cpu().numpy()
    faces = model.faces

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(mesh)

    with viewer.render_lock:
        if mesh_node is not None:
            scene.remove_node(mesh_node)

        mesh_node = scene.add(render_mesh)


    time.sleep(1/30)

print("Done.")