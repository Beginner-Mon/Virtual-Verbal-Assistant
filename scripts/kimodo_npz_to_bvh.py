"""
Kimodo NPZ -> BVH Converter
============================
Converts Kimodo-format motion NPZ (local_rot_mats + root_positions)
to standard BVH for VRM visualization.

Input NPZ keys:
    local_rot_mats   (N, 22, 3, 3)  float32
    root_positions   (N, 3)         float32 — meters
    mocap_framerate  scalar         int (default: 30)

22 SMPL-X body joints. Mirror fix applied (same as DART converter).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import argparse

# ── 22-joint SMPL-X body skeleton ────────────────────────────────────
KIMODO_JOINTS = [
    ( 0, "Hips",            -1, (  0.00,  0.00,  0.00)),
    ( 1, "LeftUpperLeg",     0, (  8.00,  0.00,  0.00)),
    ( 2, "RightUpperLeg",    0, ( -8.00,  0.00,  0.00)),
    ( 3, "Spine",            0, (  0.00,  8.00,  0.00)),
    ( 4, "LeftLowerLeg",     1, (  0.00,-42.00,  0.00)),
    ( 5, "RightLowerLeg",    2, (  0.00,-42.00,  0.00)),
    ( 6, "Spine1",           3, (  0.00,  9.00,  0.00)),
    ( 7, "LeftFoot",         4, (  0.00,-42.00,  0.00)),
    ( 8, "RightFoot",        5, (  0.00,-42.00,  0.00)),
    ( 9, "Spine2",           6, (  0.00,  9.00,  0.00)),
    (10, "LeftToe",          7, (  0.00, -8.00, 10.00)),
    (11, "RightToe",         8, (  0.00, -8.00, 10.00)),
    (12, "Neck",             9, (  0.00, 20.00,  0.00)),
    (13, "LeftShoulder",     9, (  5.00, 18.00,  0.00)),
    (14, "RightShoulder",    9, ( -5.00, 18.00,  0.00)),
    (15, "Head",            12, (  0.00, 10.00,  0.00)),
    (16, "LeftUpperArm",    13, ( 15.00,  0.00,  0.00)),
    (17, "RightUpperArm",   14, (-15.00,  0.00,  0.00)),
    (18, "LeftLowerArm",    16, ( 28.00,  0.00,  0.00)),
    (19, "RightLowerArm",   17, (-28.00,  0.00,  0.00)),
    (20, "LeftHand",        18, ( 25.00,  0.00,  0.00)),
    (21, "RightHand",       19, (-25.00,  0.00,  0.00)),
]

NUM_JOINTS = len(KIMODO_JOINTS)
MIRROR_MAT = np.diag([-1.0, 1.0, 1.0])

# Orientation correction: X=90deg + Z=180deg (R_z(180) @ R_x(90))
_ROT_CORR = np.array([[-1., 0., 0.],
                       [ 0., 0., 1.],
                       [ 0., 1., 0.]])

# Spine damping
_SPINE_DAMPING = {3: 0.35, 6: 0.40, 9: 0.40}

# Bone name swap to compensate for X-mirror in _ROT_CORR
def _swap_lr(name):
    return name.replace("Left", "_T_").replace("Right", "Left").replace("_T_", "Right")


# ── Helpers ──────────────────────────────────────────────────────────

def mirror_rotmat(mat):
    return MIRROR_MAT @ mat @ MIRROR_MAT


def rotmat_to_euler_ZXY(mat):
    mirrored = mirror_rotmat(mat)
    return R.from_matrix(mirrored).as_euler('ZXY', degrees=True)


def build_traversal_order():
    children_map = {i: [] for i in range(-1, NUM_JOINTS)}
    for idx, name, parent, offset in KIMODO_JOINTS:
        children_map[parent].append(idx)
    order = []
    def dfs(idx):
        order.append(idx)
        for child in children_map[idx]:
            dfs(child)
    for root in children_map[-1]:
        dfs(root)
    return order


def build_hierarchy_text():
    children_map = {i: [] for i in range(NUM_JOINTS)}
    name_map = {}
    offset_map = {}
    for idx, name, parent, offset in KIMODO_JOINTS:
        name_map[idx] = _swap_lr(name)
        offset_map[idx] = offset
        if parent >= 0:
            children_map[parent].append(idx)

    lines = ["HIERARCHY"]

    def write_joint(idx, indent):
        pad = "\t" * indent
        name = name_map[idx]
        ox, oy, oz = offset_map[idx]
        if indent == 0:
            lines.append(f"ROOT {name}")
        else:
            lines.append(f"{pad}JOINT {name}")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}\tOFFSET {ox:.4f} {oy:.4f} {oz:.4f}")
        if indent == 0:
            lines.append(f"{pad}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}\tCHANNELS 3 Zrotation Xrotation Yrotation")
        kids = children_map.get(idx, [])
        if kids:
            for child in kids:
                write_joint(child, indent + 1)
        else:
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET 0.0000 5.0000 0.0000")
            lines.append(f"{pad}\t}}")
        lines.append(f"{pad}}}")

    roots = [i for i, n, p, o in KIMODO_JOINTS if p == -1]
    for root in roots:
        write_joint(root, 0)
    return "\n".join(lines)


# ── Input formats ────────────────────────────────────────────────────

def load_motion(npz_path):
    """Normalise the NPZ layouts we actually get into (rot_mats, root_pos, fps).

    Two shapes are in circulation and they are NOT interchangeable:

      Kimodo native   local_rot_mats (N,22,3,3) + root_positions (N,3)
      AMASS / SMPL-X  poses (N,165) axis-angle + trans (N,3)

    The AMASS layout is what the DART-era exports use, and two of the clips
    shipped in the frontend are still in it — they were converted by a different
    tool, which is why they never went through this script's fixes.
    `poses[:, :66]` is root orientation followed by the 21 body joints, i.e.
    exactly the 22 joints this skeleton uses; the remaining columns are hands
    and face, which have no BVH counterpart here.
    """
    data = np.load(npz_path, allow_pickle=True)
    fps = int(data['mocap_framerate']) if 'mocap_framerate' in data.files else None

    if 'local_rot_mats' in data.files:
        return data['local_rot_mats'], data['root_positions'], fps, 'kimodo'

    if 'poses' in data.files and 'trans' in data.files:
        poses = np.asarray(data['poses'], dtype=float)
        rotvecs = poses[:, : NUM_JOINTS * 3].reshape(-1, 3)
        mats = R.from_rotvec(rotvecs).as_matrix().reshape(len(poses), NUM_JOINTS, 3, 3)
        return mats, np.asarray(data['trans'], dtype=float), fps, 'amass'

    raise KeyError(
        f"{npz_path}: expected either 'local_rot_mats'+'root_positions' (Kimodo) "
        f"or 'poses'+'trans' (AMASS/SMPL-X); found {sorted(data.files)}"
    )


# ── Grounding ────────────────────────────────────────────────────────

def joint_rotations(local_rot_mats, f, smplx_idx):
    """The rotation actually written to BVH for this joint on this frame.

    Must stay in lockstep with the writer below: spine damping first, then the
    mirror fix. FK done on anything else would ground the wrong skeleton.
    """
    mat = local_rot_mats[f, smplx_idx]
    factor = _SPINE_DAMPING.get(smplx_idx, 1.0)
    if factor < 1.0:
        rotvec = R.from_matrix(mat).as_rotvec()
        mat = R.from_rotvec(rotvec * factor).as_matrix()
    return mirror_rotmat(mat)


# ── Converter ────────────────────────────────────────────────────────

def convert_npz_to_bvh(npz_path, bvh_path, framerate=30):
    print(f"Loading: {npz_path}")
    local_rot_mats, root_positions, file_fps, layout = load_motion(npz_path)
    if file_fps:
        framerate = file_fps

    num_frames = local_rot_mats.shape[0]
    frame_time = 1.0 / framerate
    print(f"  Format:      {layout} -> 22-joint")
    print(f"  Frames:      {num_frames}")
    print(f"  Framerate:   {framerate} FPS")

    traversal_order = build_traversal_order()
    hierarchy = build_hierarchy_text()

    motion_lines = ["MOTION", f"Frames: {num_frames}", f"Frame Time: {frame_time:.6f}"]

    for f in range(num_frames):
        frame_values = []

        # Root translation: correction + meters->cm + mirror.
        # NOTE: height lives in the Z channel here, not Y — the frontend reads
        # these files with `swapYandZ`. Do NOT "ground" this file: the retarget
        # anchors frame 0 to the VRM's rest hip height and applies deltas, so
        # absolute height in the BVH is discarded. Grounding belongs at the
        # consumer (see ECA_UI/frontend/src/lib/groundClamp.ts).
        corr = _ROT_CORR @ root_positions[f] * 100.0
        frame_values.extend([-corr[0], corr[1], corr[2]])

        # Root rotation: correction + mirror
        root_rot = _ROT_CORR @ local_rot_mats[f, 0]
        rz, rx, ry = rotmat_to_euler_ZXY(root_rot)
        frame_values.extend([rz, rx, ry])

        # Child joints: mirror fix + optional spine damping
        for bvh_pos in range(1, NUM_JOINTS):
            smplx_idx = traversal_order[bvh_pos]
            rz, rx, ry = R.from_matrix(
                joint_rotations(local_rot_mats, f, smplx_idx)
            ).as_euler('ZXY', degrees=True)
            frame_values.extend([rz, rx, ry])

        motion_lines.append(" ".join(f"{v:.6f}" for v in frame_values))

    print(f"Writing: {bvh_path}")
    with open(bvh_path, 'w') as f_out:
        f_out.write(hierarchy)
        f_out.write("\n")
        f_out.write("\n".join(motion_lines))
        f_out.write("\n")
    print(f"Done! {os.path.getsize(bvh_path) / 1024:.1f} KB")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Kimodo NPZ motion to BVH.")
    parser.add_argument("npz_path", help="Input .npz file")
    parser.add_argument("bvh_path", nargs="?", help="Output .bvh path")
    parser.add_argument("--fps", type=int, default=30, help="FPS (default 30)")
    args = parser.parse_args()

    npz_path = args.npz_path
    bvh_path = args.bvh_path or os.path.splitext(npz_path)[0] + ".bvh"
    if not os.path.exists(npz_path):
        print(f"Error: file not found: {npz_path}")
        return 1

    convert_npz_to_bvh(npz_path, bvh_path, args.fps)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
