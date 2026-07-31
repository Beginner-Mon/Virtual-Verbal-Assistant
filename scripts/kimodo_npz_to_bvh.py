"""
NPZ → BVH Converter (Fixed)
=============================
Converts Kimodo/DART-format motion NPZ to correct BVH.

Key fixes over the previous version:
1. Uses ACTUAL bone offsets extracted from the posed_joints data
   instead of generic hardcoded values
2. No artificial spine damping
3. No upper arm Y+180° hack
4. Correct mirror fix: SMPL-X has left=+X, BVH convention has right=+X
5. Proper root translation handling

Input NPZ keys used:
    local_rot_mats   (N, 22, 3, 3)  float32 - local rotation matrices
    root_positions   (N, 3)         float32 - meters, Y-up
    global_rot_mats  (N, 22, 3, 3)  float32 - for extracting bone offsets
    posed_joints     (N, 22, 3)     float32 - for extracting bone offsets
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import json
import struct
import argparse

# ── 22-joint SMPL body skeleton (index, name, parent) ─────────────────
SMPL_JOINTS = [
    ( 0, "Hips",           -1),
    ( 1, "LeftUpperLeg",    0),
    ( 2, "RightUpperLeg",   0),
    ( 3, "Spine",           0),
    ( 4, "LeftLowerLeg",    1),
    ( 5, "RightLowerLeg",   2),
    ( 6, "Spine1",          3),
    ( 7, "LeftFoot",        4),
    ( 8, "RightFoot",       5),
    ( 9, "Spine2",          6),
    (10, "LeftToe",         7),
    (11, "RightToe",        8),
    (12, "Neck",            9),
    (13, "LeftShoulder",    9),
    (14, "RightShoulder",   9),
    (15, "Head",           12),
    (16, "LeftUpperArm",   13),
    (17, "RightUpperArm",  14),
    (18, "LeftLowerArm",   16),
    (19, "RightLowerArm",  17),
    (20, "LeftHand",       18),
    (21, "RightHand",      19),
]

NUM_JOINTS = len(SMPL_JOINTS)

# Mirror matrix: flip X axis (SMPL-X: left=+X → BVH: right=+X)
MIRROR_MAT = np.diag([-1.0, 1.0, 1.0])

# Spine joints that form the chain: Spine(3) → Spine1(6) → Spine2(9)
# VRM models typically only use 2 spine bones, so we redistribute
# the total rotation evenly across the 3 SMPL spine joints.
SPINE_JOINT_INDICES = [3, 6, 9]

# Damping factor for the first Spine joint (0.0 = no rotation, 1.0 = full).
# Lower values reduce upper-body swing when retargeted to VRM.
SPINE_DAMPING = 0.5


# ── Helpers ──────────────────────────────────────────────────────────

def mirror_rotmat(mat):
    """Mirror a rotation matrix across YZ plane (negate X axis)."""
    return MIRROR_MAT @ mat @ MIRROR_MAT


def rotmat_to_euler_ZXY(mat):
    """Convert rotation matrix to ZXY Euler angles (degrees), with mirror fix."""
    mirrored = mirror_rotmat(mat)
    return R.from_matrix(mirrored).as_euler('ZXY', degrees=True)


def damp_rotation(mat, factor):
    """
    Scale a rotation matrix by a factor using SLERP from identity.
    factor=1.0 returns the original rotation, factor=0.0 returns identity.
    """
    if factor >= 1.0:
        return mat
    if factor <= 0.0:
        return np.eye(3)
    rotvec = R.from_matrix(mat).as_rotvec()
    return R.from_rotvec(rotvec * factor).as_matrix()


def get_vrm_spine_count(vrm_path):
    """
    Parse a VRM file (GLB format) and count how many spine bones are mapped.
    VRM humanoid spine bones: spine, chest, upperChest.
    Returns the count (1-3) or 3 if the file can't be parsed.
    """
    VRM_SPINE_BONES = ['spine', 'chest', 'upperChest']
    try:
        with open(vrm_path, 'rb') as f:
            # GLB header: magic(4) + version(4) + length(4)
            magic = f.read(4)
            if magic != b'glTF':
                print(f"  Warning: {vrm_path} is not a valid GLB file")
                return 3
            _version, _length = struct.unpack('<II', f.read(8))

            # First chunk: JSON
            chunk_length, chunk_type = struct.unpack('<II', f.read(8))
            if chunk_type != 0x4E4F534A:  # "JSON"
                print(f"  Warning: first chunk is not JSON")
                return 3
            json_data = json.loads(f.read(chunk_length))

        # Try VRM 1.0 format: extensions.VRMC_vrm.humanoid.humanBones
        vrmc = json_data.get('extensions', {}).get('VRMC_vrm', {})
        human_bones = vrmc.get('humanoid', {}).get('humanBones', {})
        if human_bones:
            count = sum(1 for bone in VRM_SPINE_BONES if bone in human_bones)
            return max(count, 1)

        # Try VRM 0.x format: extensions.VRM.humanoid.humanBones (array)
        vrm0 = json_data.get('extensions', {}).get('VRM', {})
        human_bones_list = vrm0.get('humanoid', {}).get('humanBones', [])
        if human_bones_list:
            mapped = {entry['bone'] for entry in human_bones_list if 'bone' in entry}
            count = sum(1 for bone in VRM_SPINE_BONES if bone in mapped)
            return max(count, 1)

        print(f"  Warning: could not find humanoid bone mapping in VRM")
        return 3
    except Exception as e:
        print(f"  Warning: failed to parse VRM file: {e}")
        return 3


def extract_bone_offsets(posed_joints, global_rot_mats):
    """
    Extract rest-pose bone offsets from the NPZ data.
    
    For each joint j with parent p:
        offset[j] = inv(global_rot[p]) @ (pos[j] - pos[p])
    
    These offsets are constant across all frames (rigid skeleton),
    so we use frame 0.
    
    Returns offsets in meters.
    """
    offsets = np.zeros((NUM_JOINTS, 3))
    # Root offset is (0,0,0) — position comes from root_positions
    for idx, name, parent in SMPL_JOINTS:
        if parent >= 0:
            diff = posed_joints[0, idx] - posed_joints[0, parent]
            offsets[idx] = global_rot_mats[0, parent].T @ diff

    # Clean up spine chain offsets: zero out X and Z components
    # SMPL has a natural spinal S-curve encoded in the bone offsets,
    # but BVH rigs expect a straight vertical spine in rest pose.
    SPINE_CHAIN = {3, 6, 9, 12}  # Spine, Spine1, Spine2, Neck
    for idx in SPINE_CHAIN:
        offsets[idx, 0] = 0.0  # zero X
        offsets[idx, 2] = 0.0  # zero Z

    return offsets


def build_traversal_order():
    """Build depth-first traversal order for BVH output."""
    children_map = {i: [] for i in range(-1, NUM_JOINTS)}
    for idx, name, parent in SMPL_JOINTS:
        children_map[parent].append(idx)
    order = []
    def dfs(idx):
        order.append(idx)
        for child in children_map[idx]:
            dfs(child)
    for root in children_map[-1]:
        dfs(root)
    return order


def build_hierarchy_text(offsets_cm):
    """Build BVH HIERARCHY section using actual bone offsets."""
    children_map = {i: [] for i in range(NUM_JOINTS)}
    name_map = {}
    for idx, name, parent in SMPL_JOINTS:
        name_map[idx] = name
        if parent >= 0:
            children_map[parent].append(idx)

    lines = ["HIERARCHY"]

    def write_joint(idx, indent):
        pad = "\t" * indent
        name = name_map[idx]
        ox, oy, oz = offsets_cm[idx]
        
        # Mirror the X offset for BVH convention
        ox = -ox
        
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
            # End site: small offset along the bone direction
            lines.append(f"{pad}\tEnd Site")
            lines.append(f"{pad}\t{{")
            lines.append(f"{pad}\t\tOFFSET 0.0000 5.0000 0.0000")
            lines.append(f"{pad}\t}}")
        lines.append(f"{pad}}}")

    roots = [i for i, n, p in SMPL_JOINTS if p == -1]
    for root in roots:
        write_joint(root, 0)
    return "\n".join(lines)


# ── Converter ────────────────────────────────────────────────────────

def convert_npz_to_bvh(npz_path, bvh_path, framerate=30, spine_damping=SPINE_DAMPING, vrm_path=None):
    print(f"Loading: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    local_rot_mats = data['local_rot_mats']     # (N, 22, 3, 3)
    root_positions = data['root_positions']       # (N, 3)
    global_rot_mats = data['global_rot_mats']     # (N, 22, 3, 3)
    posed_joints = data['posed_joints']           # (N, 22, 3)
    
    if 'mocap_framerate' in data:
        framerate = int(data['mocap_framerate'])

    num_frames = local_rot_mats.shape[0]
    frame_time = 1.0 / framerate
    
    # Detect VRM spine bone count to decide whether damping is needed
    apply_damping = False
    if vrm_path:
        vrm_spine_count = get_vrm_spine_count(vrm_path)
        print(f"  VRM model:   {os.path.basename(vrm_path)}")
        print(f"  VRM spines:  {vrm_spine_count} (spine/chest/upperChest)")
        if vrm_spine_count < 3:
            apply_damping = True
            print(f"  Spine damp:  {spine_damping:.2f} (VRM has <3 spine bones, redistributing excess)")
        else:
            print(f"  Spine damp:  OFF (VRM has all 3 spine bones)")
    else:
        print(f"  VRM model:   not specified (no damping applied)")

    # Extract actual bone offsets from the skeleton data
    offsets_m = extract_bone_offsets(posed_joints, global_rot_mats)
    offsets_cm = offsets_m * 100.0
    
    print(f"  Frames:      {num_frames}")
    print(f"  Framerate:   {framerate} FPS")
    print(f"  Duration:    {num_frames / framerate:.2f}s")
    print(f"  Skeleton:    Extracted from posed_joints (actual proportions)")
    
    # Print extracted offsets for verification
    print(f"\n  Bone offsets (cm):")
    for idx, name, parent in SMPL_JOINTS:
        if parent >= 0:
            ox, oy, oz = offsets_cm[idx]
            print(f"    {name:20s}: ({ox:7.2f}, {oy:7.2f}, {oz:7.2f})")

    traversal_order = build_traversal_order()
    hierarchy = build_hierarchy_text(offsets_cm)

    motion_lines = ["MOTION", f"Frames: {num_frames}", f"Frame Time: {frame_time:.6f}"]

    for f in range(num_frames):
        frame_values = []

        # Root translation: meters → cm, X negated for mirror fix
        tx, ty, tz = root_positions[f] * 100.0
        frame_values.extend([-tx, ty, tz])

        frame_rots = local_rot_mats[f].copy()  # (22, 3, 3)

        if apply_damping:
            # Damp the first Spine joint rotation to reduce upper-body swing
            # when retargeted to VRM (which has fewer spine bones).
            # The excess rotation is redistributed to Spine1 and Spine2.
            spine_idx = SPINE_JOINT_INDICES[0]   # joint 3 = Spine
            spine1_idx = SPINE_JOINT_INDICES[1]   # joint 6 = Spine1
            spine2_idx = SPINE_JOINT_INDICES[2]   # joint 9 = Spine2

            original_spine = frame_rots[spine_idx].copy()
            damped_spine = damp_rotation(original_spine, spine_damping)

            # The residual rotation that was removed from Spine
            # residual = inv(damped) @ original
            residual = damped_spine.T @ original_spine

            # Split the residual evenly between Spine1 and Spine2
            residual_rotvec = R.from_matrix(residual).as_rotvec()
            half_residual = R.from_rotvec(residual_rotvec * 0.5).as_matrix()

            frame_rots[spine_idx] = damped_spine
            frame_rots[spine1_idx] = half_residual @ frame_rots[spine1_idx]
            frame_rots[spine2_idx] = half_residual @ frame_rots[spine2_idx]

        # All joints in BVH depth-first traversal order
        for bvh_pos in range(NUM_JOINTS):
            smplx_idx = traversal_order[bvh_pos]
            mat = frame_rots[smplx_idx]
            rz, rx, ry = rotmat_to_euler_ZXY(mat)
            frame_values.extend([rz, rx, ry])

        motion_lines.append(" ".join(f"{v:.6f}" for v in frame_values))

    print(f"\nWriting: {bvh_path}")
    with open(bvh_path, 'w') as f_out:
        f_out.write(hierarchy)
        f_out.write("\n")
        f_out.write("\n".join(motion_lines))
        f_out.write("\n")
    print(f"Done! {os.path.getsize(bvh_path) / 1024:.1f} KB")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Kimodo/DART NPZ motion to BVH (fixed version).")
    parser.add_argument("npz_path", help="Input .npz file")
    parser.add_argument("bvh_path", nargs="?", help="Output .bvh path")
    parser.add_argument("--fps", type=int, default=30, help="FPS (default 30)")
    parser.add_argument("--vrm", dest="vrm_path", default=None,
                        help="Path to the target VRM model. If provided, the script detects "
                             "the number of spine bones and applies damping only if <3.")
    parser.add_argument("--spine-damping", type=float, default=SPINE_DAMPING,
                        help=f"Damping factor for the first Spine joint (0.0-1.0, default {SPINE_DAMPING}). "
                             "Only used when VRM has <3 spine bones.")
    args = parser.parse_args()

    npz_path = args.npz_path
    bvh_path = args.bvh_path or os.path.splitext(npz_path)[0] + ".bvh"
    if not os.path.exists(npz_path):
        print(f"Error: file not found: {npz_path}")
        return 1
    if args.vrm_path and not os.path.exists(args.vrm_path):
        print(f"Error: VRM file not found: {args.vrm_path}")
        return 1

    convert_npz_to_bvh(npz_path, bvh_path, args.fps, args.spine_damping, args.vrm_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


