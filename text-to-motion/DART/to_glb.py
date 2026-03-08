import argparse

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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    data = np.load(args.npz_path, allow_pickle=True)

    poses = data["poses"]
    trans = data["trans"]
    betas = data["betas"]
    num_frames = poses.shape[0]

    model = smplx.create(
        args.model_path,
        model_type="smplx",
        gender=args.gender,
        use_pca=False,
    ).to(device)

    betas = torch.tensor(betas).float().unsqueeze(0)
    scene = trimesh.Scene()

    for frame_idx in range(num_frames):
        pose = torch.tensor(poses[frame_idx]).float().unsqueeze(0)
        translation = torch.tensor(trans[frame_idx]).float().unsqueeze(0)

        with torch.no_grad():
            out = model(
                betas=betas,
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
        mesh = trimesh.Trimesh(vertices, faces)
        scene.add_geometry(mesh, node_name=f"frame_{frame_idx}")

    glb_bytes = trimesh.exchange.gltf.export_glb(scene)
    with open(args.out_path, "wb") as file:
        file.write(glb_bytes)

    print("Saved:", args.out_path)


if __name__ == "__main__":
    main()
