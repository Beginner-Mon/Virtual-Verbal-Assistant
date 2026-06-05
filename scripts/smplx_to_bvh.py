"""
SMPL-X NPZ → BVH Converter v3
================================
Fixes applied:
  v1 → v2: BVH motion data written in correct depth-first traversal order.
  v2 → v3: Mirror fix — SMPL-X X-axis points LEFT, BVH X-axis points RIGHT.
            Negates X translation + mirrors all rotations across YZ plane
            so left/right hands and legs are no longer swapped.

Usage:
    python smplx_to_bvh.py input.npz output.bvh
    python smplx_to_bvh.py input.npz
    python smplx_to_bvh.py input.npz output.bvh --diagnose

Requirements:
    pip install numpy scipy
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SMPL-X skeleton: (index, joint_name, parent_index, offset_cm)
# ─────────────────────────────────────────────────────────────────────────────
SMPLX_JOINTS = [
    ( 0, "Hips",                   -1, (  0.00,  0.00,  0.00)),
    ( 1, "LeftUpperLeg",            0, (  8.00,  0.00,  0.00)),
    ( 2, "RightUpperLeg",           0, ( -8.00,  0.00,  0.00)),
    ( 3, "Spine",                   0, (  0.00,  8.00,  0.00)),
    ( 4, "LeftLowerLeg",            1, (  0.00,-42.00,  0.00)),
    ( 5, "RightLowerLeg",           2, (  0.00,-42.00,  0.00)),
    ( 6, "Spine1",                  3, (  0.00,  9.00,  0.00)),
    ( 7, "LeftFoot",                4, (  0.00,-42.00,  0.00)),
    ( 8, "RightFoot",               5, (  0.00,-42.00,  0.00)),
    ( 9, "Spine2",                  6, (  0.00,  9.00,  0.00)),
    (10, "LeftToe",                 7, (  0.00, -8.00, 10.00)),
    (11, "RightToe",                8, (  0.00, -8.00, 10.00)),
    (12, "Neck",                    9, (  0.00, 20.00,  0.00)),
    (13, "LeftShoulder",            9, (  5.00, 18.00,  0.00)),
    (14, "RightShoulder",           9, ( -5.00, 18.00,  0.00)),
    (15, "Head",                   12, (  0.00, 10.00,  0.00)),
    (16, "LeftUpperArm",           13, ( 15.00,  0.00,  0.00)),
    (17, "RightUpperArm",          14, (-15.00,  0.00,  0.00)),
    (18, "LeftLowerArm",           16, ( 28.00,  0.00,  0.00)),
    (19, "RightLowerArm",          17, (-28.00,  0.00,  0.00)),
    (20, "LeftHand",               18, ( 25.00,  0.00,  0.00)),
    (21, "RightHand",              19, (-25.00,  0.00,  0.00)),
    (22, "Jaw",                    15, (  0.00, -4.00,  5.00)),
    (23, "LeftEye",                15, (  3.00,  5.00,  7.00)),
    (24, "RightEye",               15, ( -3.00,  5.00,  7.00)),
    (25, "LeftIndexProximal",      20, (  5.00, -1.00,  1.00)),
    (26, "LeftIndexIntermediate",  25, (  4.00,  0.00,  0.00)),
    (27, "LeftIndexDistal",        26, (  3.00,  0.00,  0.00)),
    (28, "LeftMiddleProximal",     20, (  5.00,  0.00,  0.00)),
    (29, "LeftMiddleIntermediate", 28, (  4.00,  0.00,  0.00)),
    (30, "LeftMiddleDistal",       29, (  3.00,  0.00,  0.00)),
    (31, "LeftLittleProximal",     20, (  5.00,  1.50,  0.00)),
    (32, "LeftLittleIntermediate", 31, (  3.00,  0.00,  0.00)),
    (33, "LeftLittleDistal",       32, (  2.00,  0.00,  0.00)),
    (34, "LeftRingProximal",       20, (  5.00,  0.80,  0.00)),
    (35, "LeftRingIntermediate",   34, (  4.00,  0.00,  0.00)),
    (36, "LeftRingDistal",         35, (  3.00,  0.00,  0.00)),
    (37, "LeftThumbProximal",      20, (  3.00, -2.00,  2.00)),
    (38, "LeftThumbIntermediate",  37, (  3.00,  0.00,  0.00)),
    (39, "LeftThumbDistal",        38, (  2.50,  0.00,  0.00)),
    (40, "RightIndexProximal",     21, ( -5.00, -1.00,  1.00)),
    (41, "RightIndexIntermediate", 40, ( -4.00,  0.00,  0.00)),
    (42, "RightIndexDistal",       41, ( -3.00,  0.00,  0.00)),
    (43, "RightMiddleProximal",    21, ( -5.00,  0.00,  0.00)),
    (44, "RightMiddleIntermediate",43, ( -4.00,  0.00,  0.00)),
    (45, "RightMiddleDistal",      44, ( -3.00,  0.00,  0.00)),
    (46, "RightLittleProximal",    21, ( -5.00,  1.50,  0.00)),
    (47, "RightLittleIntermediate",46, ( -3.00,  0.00,  0.00)),
    (48, "RightLittleDistal",      47, ( -2.00,  0.00,  0.00)),
    (49, "RightRingProximal",      21, ( -5.00,  0.80,  0.00)),
    (50, "RightRingIntermediate",  49, ( -4.00,  0.00,  0.00)),
    (51, "RightRingDistal",        50, ( -3.00,  0.00,  0.00)),
    (52, "RightThumbProximal",     21, ( -3.00, -2.00,  2.00)),
    (53, "RightThumbIntermediate", 52, ( -3.00,  0.00,  0.00)),
    (54, "RightThumbDistal",       53, ( -2.50,  0.00,  0.00)),
]

NUM_JOINTS = len(SMPLX_JOINTS)

# Mirror matrix: flip X axis to convert SMPL-X (X=left) → BVH (X=right)
# Applied as: R_bvh = M @ R_smplx @ M
MIRROR_MAT = np.diag([-1.0, 1.0, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mirror_rotvec(rotvec):
    """Mirror a rotation across the YZ plane (flip X axis)."""
    if np.linalg.norm(rotvec) < 1e-8:
        return rotvec
    mat = R.from_rotvec(rotvec).as_matrix()
    return R.from_matrix(MIRROR_MAT @ mat @ MIRROR_MAT).as_rotvec()


def rotvec_to_euler_ZXY(rotvec):
    """Axis-angle → ZXY Euler degrees (BVH convention), with mirror applied."""
    mirrored = mirror_rotvec(rotvec)
    ez, ex, ey = R.from_rotvec(mirrored).as_euler('ZXY', degrees=True)
    return ez, ex, ey


def build_traversal_order():
    """
    BVH motion data must be in depth-first traversal order of the skeleton,
    NOT in SMPL-X joint index order.
    Returns: traversal_order[bvh_position] = smplx_joint_index
    """
    children = {i: [] for i in range(-1, NUM_JOINTS)}
    for idx, name, parent, offset in SMPLX_JOINTS:
        children[parent].append(idx)

    order = []
    def dfs(idx):
        order.append(idx)
        for child in children[idx]:
            dfs(child)

    for root_idx in children[-1]:
        dfs(root_idx)
    return order


def build_hierarchy_text():
    """Build the HIERARCHY section of the BVH file."""
    children_map = {i: [] for i in range(NUM_JOINTS)}
    name_map     = {}
    offset_map   = {}
    for idx, name, parent, offset in SMPLX_JOINTS:
        name_map[idx]   = name
        offset_map[idx] = offset
        if parent >= 0:
            children_map[parent].append(idx)

    lines = ["HIERARCHY"]

    def write_joint(idx, indent):
        pad  = "\t" * indent
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

        kids = children_map[idx]
        if kids:
            for child in kids:
                write_joint(child, indent + 1)
        else:
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET 0.0000 5.0000 0.0000")
            lines.append(f"{pad}\t}}")

        lines.append(f"{pad}}}")

    roots = [idx for idx, name, parent, offset in SMPLX_JOINTS if parent == -1]
    for root in roots:
        write_joint(root, 0)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_diagnostics(npz_path, data, traversal_order):
    poses     = data['poses']
    trans     = data['trans']
    framerate = int(data['mocap_framerate'])

    print("\nDiagnostics")
    print("-----------")
    print(f"NPZ path:       {npz_path}")
    print(f"Poses shape:    {poses.shape}")
    print(f"Trans shape:    {trans.shape}")
    print(f"Framerate:      {framerate} FPS")
    print(f"Frames:         {poses.shape[0]}")
    print(f"Joint count:    {NUM_JOINTS}")
    print(f"Mirror fix:     ON (SMPL-X X=left -> BVH X=right)")

    print("\nTraversal order (BVH position -> SMPL-X joint name)")
    for bvh_pos, smplx_idx in enumerate(traversal_order):
        joint_name = SMPLX_JOINTS[smplx_idx][1]
        print(f"  {bvh_pos:02d}: {smplx_idx:02d} {joint_name}")

    if poses.shape[0] > 0:
        print("\nFirst frame summary (after mirror fix)")
        tx, ty, tz = trans[0] * 100.0
        print(f"  Root translation cm: ({-tx:.3f}, {ty:.3f}, {tz:.3f})  [X negated]")

        rz, rx, ry = rotvec_to_euler_ZXY(poses[0, 0:3])
        print(f"  Root rotation ZXY:   ({rz:.3f}, {rx:.3f}, {ry:.3f})")

        sample_count = min(8, NUM_JOINTS)
        print("  First joints (ZXY degrees, mirrored):")
        for bvh_pos in range(sample_count):
            smplx_idx  = traversal_order[bvh_pos]
            joint_name = SMPLX_JOINTS[smplx_idx][1]
            aa = poses[0, smplx_idx*3:(smplx_idx+1)*3]
            rz, rx, ry = rotvec_to_euler_ZXY(aa)
            print(f"    {bvh_pos:02d} {joint_name:<20} -> ({rz:8.3f}, {rx:8.3f}, {ry:8.3f})")

    print("\nExpected BVH motion values per frame:")
    print(f"  {3 + 3 + (NUM_JOINTS - 1) * 3} values")


# ─────────────────────────────────────────────────────────────────────────────
# Main converter
# ─────────────────────────────────────────────────────────────────────────────

def convert_npz_to_bvh(npz_path, bvh_path):
    print(f"Loading: {npz_path}")
    data      = np.load(npz_path, allow_pickle=True)
    poses     = data['poses']        # (N, 165) axis-angle
    trans     = data['trans']        # (N, 3)   metres
    framerate = int(data['mocap_framerate'])

    num_frames = poses.shape[0]
    frame_time = 1.0 / framerate

    print(f"  Frames:    {num_frames}")
    print(f"  Framerate: {framerate} FPS")
    print(f"  Duration:  {num_frames / framerate:.2f}s")
    print(f"  Mirror fix: ON")

    traversal_order = build_traversal_order()
    print(f"  Traversal order built: {len(traversal_order)} joints")

    print("Building skeleton hierarchy...")
    hierarchy = build_hierarchy_text()

    print("Converting rotations (depth-first order, mirrored)...")
    motion_lines = [
        "MOTION",
        f"Frames: {num_frames}",
        f"Frame Time: {frame_time:.6f}",
    ]

    for frame_idx in range(num_frames):
        frame_values = []

        # Root translation: metres → cm, X negated for mirror fix
        tx, ty, tz = trans[frame_idx] * 100.0
        frame_values.extend([-tx, ty, tz])

        # Root rotation (smplx index 0), mirrored
        rz, rx, ry = rotvec_to_euler_ZXY(poses[frame_idx, 0:3])
        frame_values.extend([rz, rx, ry])

        # All other joints in BVH traversal order, mirrored
        for bvh_pos in range(1, NUM_JOINTS):
            smplx_idx  = traversal_order[bvh_pos]
            aa         = poses[frame_idx, smplx_idx*3:(smplx_idx+1)*3]
            rz, rx, ry = rotvec_to_euler_ZXY(aa)
            frame_values.extend([rz, rx, ry])

        motion_lines.append(" ".join(f"{v:.6f}" for v in frame_values))

    print(f"Writing: {bvh_path}")
    with open(bvh_path, 'w') as f:
        f.write(hierarchy)
        f.write("\n")
        f.write("\n".join(motion_lines))
        f.write("\n")

    size_kb = os.path.getsize(bvh_path) / 1024
    print(f"Done! Output: {size_kb:.1f} KB")
    print()
    print("Next steps:")
    print("  1. Import the .bvh into Blender (File → Import → BVH)")
    print("  2. Retarget onto your VRM using Auto-Rig Pro or Rokoko")
    print("  3. Export as .vrma or bake into .glb for Three.js")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def resolve_default_npz_path() -> str | None:
    """Return the bundled motion NPZ path if it exists."""
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent / "ECA_UI" / "frontend" / "src" / "asset" / "motion_b28e8284-328.npz"
    return str(candidate) if candidate.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert SMPL-X NPZ motion to BVH.")
    parser.add_argument("npz_path", nargs="?", help="Input .npz file")
    parser.add_argument("bvh_path", nargs="?", help="Output .bvh path (defaults to input basename)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print joint-order and first-frame diagnostics before writing")
    args = parser.parse_args()

    npz_path = args.npz_path or resolve_default_npz_path()
    if not npz_path:
        parser.print_help()
        print("\nNo input .npz was provided and the bundled default motion file was not found.")
        return 1

    bvh_path = args.bvh_path if args.bvh_path else os.path.splitext(npz_path)[0] + ".bvh"

    if not os.path.exists(npz_path):
        print(f"Error: file not found: {npz_path}")
        return 1

    print(f"[main] Input:  {npz_path}")
    print(f"[main] Output: {bvh_path}")
    if args.diagnose:
        print("[main] Diagnostics enabled")
        data = np.load(npz_path, allow_pickle=True)
        traversal_order = build_traversal_order()
        print_diagnostics(npz_path, data, traversal_order)

    convert_npz_to_bvh(npz_path, bvh_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())