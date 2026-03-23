#!/usr/bin/env python3
"""
npz_to_animated_glb.py
Converts an SMPL-X .npz motion capture file to a fully animated .glb
using morph-target (blend-shape) animation — no external skeleton needed.

Usage:
    python npz_to_animated_glb.py [input.npz] [output.glb]

Requirements:
    pip install smplx trimesh torch pygltflib numpy
"""

import numpy as np
import torch
import smplx
import struct
import sys
from pathlib import Path

try:
    import pygltflib
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygltflib"])
    import pygltflib

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_NPZ  = sys.argv[1] if len(sys.argv) > 1 else "test.npz"
OUTPUT_GLB = sys.argv[2] if len(sys.argv) > 2 else "animated.glb"
MODEL_PATH = "./models"
FPS        = 30          # playback speed in the browser
MAX_FRAMES = 120         # cap so file size stays reasonable (~15 MB for SMPL-X)

# ── LOAD NPZ ──────────────────────────────────────────────────────────────────
data   = np.load(INPUT_NPZ)
poses  = data["poses"]   # (T, 165)
betas  = data["betas"]   # (10,) or (1,10)
trans  = data["trans"]   # (T, 3)
gender = str(data["gender"])

total  = len(poses)
stride = max(1, total // MAX_FRAMES)
frame_indices = list(range(0, total, stride))[:MAX_FRAMES]
n_frames = len(frame_indices)
print(f"[npz→glb] {total} total poses → using {n_frames} frames  (stride={stride}, fps={FPS})")

# ── SMPL-X INFERENCE ─────────────────────────────────────────────────────────
model = smplx.create(
    model_path=MODEL_PATH, model_type="smplx",
    gender=gender, ext="pkl", use_pca=False
)

# betas can be 1-D (10,) or 2-D (1,10) — normalise to (1,10)
betas_np = betas.reshape(1, -1)[:, :10]
betas_t  = torch.tensor(betas_np).float()

def run_frame(fi: int) -> np.ndarray:
    p = poses[fi]
    t = trans[fi]
    out = model(
        betas=betas_t,
        global_orient=torch.tensor(p[0:3]).float().unsqueeze(0),
        body_pose=torch.tensor(p[3:66]).float().unsqueeze(0),
        left_hand_pose=torch.tensor(p[66:111]).float().unsqueeze(0),
        right_hand_pose=torch.tensor(p[111:156]).float().unsqueeze(0),
        jaw_pose=torch.tensor(p[156:159]).float().unsqueeze(0),
        leye_pose=torch.tensor(p[159:162]).float().unsqueeze(0),
        reye_pose=torch.tensor(p[162:165]).float().unsqueeze(0),
        transl=torch.tensor(t).float().unsqueeze(0),
    )
    return out.vertices.detach().cpu().numpy().squeeze().astype(np.float32)

print("[npz→glb] Running SMPL-X for each frame…")
all_verts = []
for i, fi in enumerate(frame_indices):
    print(f"\r  frame {i+1}/{n_frames}", end="", flush=True)
    all_verts.append(run_frame(fi))
print()

faces   = model.faces.astype(np.uint32)   # (F, 3)
n_verts = all_verts[0].shape[0]
n_faces = faces.shape[0]
base    = all_verts[0]                     # reference shape  (V, 3)
n_morph = n_frames - 1                     # one morph target per extra frame

# ── BINARY BUFFER BUILDER ────────────────────────────────────────────────────
blobs  = []   # raw bytes chunks
views  = []   # (byteOffset, byteLength, target_or_None)

def add_blob(raw: bytes, target=None) -> int:
    """Append 4-byte-aligned chunk; return its bufferView index."""
    b = bytes(raw)
    pad = (4 - len(b) % 4) % 4
    b += b'\x00' * pad
    offset = sum(len(x) for x in blobs)
    blobs.append(b)
    views.append((offset, len(b), target))
    return len(views) - 1

accessors_meta = []  # list of dicts for pygltflib.Accessor creation

def add_accessor(view_idx: int, component_type, count: int, acc_type: str,
                 min_val=None, max_val=None) -> int:
    accessors_meta.append(dict(
        bufferView=view_idx, componentType=component_type,
        count=count, type=acc_type, min=min_val, max=max_val
    ))
    return len(accessors_meta) - 1

# 1. Base vertex positions
pos_view = add_blob(base.tobytes(), pygltflib.ARRAY_BUFFER)
pos_acc  = add_accessor(pos_view, pygltflib.FLOAT, n_verts, pygltflib.VEC3,
                        base.min(axis=0).tolist(), base.max(axis=0).tolist())

# 2. Triangle indices
idx_bytes = faces.flatten().tobytes()
idx_view  = add_blob(idx_bytes, pygltflib.ELEMENT_ARRAY_BUFFER)
idx_acc   = add_accessor(idx_view, pygltflib.UNSIGNED_INT, n_faces * 3, pygltflib.SCALAR)

# 3. Morph target deltas  Δv = v_i − v_0
morph_accs = []
for i in range(1, n_frames):
    delta    = (all_verts[i] - base).astype(np.float32)
    d_view   = add_blob(delta.tobytes(), pygltflib.ARRAY_BUFFER)
    d_acc    = add_accessor(d_view, pygltflib.FLOAT, n_verts, pygltflib.VEC3,
                            delta.min(axis=0).tolist(), delta.max(axis=0).tolist())
    morph_accs.append(d_acc)

# 4. Animation timestamps  [0, 1/fps, 2/fps, …]
times   = np.array([i / FPS for i in range(n_frames)], dtype=np.float32)
t_view  = add_blob(times.tobytes())
t_acc   = add_accessor(t_view, pygltflib.FLOAT, n_frames, pygltflib.SCALAR,
                       [float(times[0])], [float(times[-1])])

# 5. Morph weights output
#    Shape (n_frames, n_morph): frame i → weight[i-1]=1, rest=0   (STEP interp)
if n_morph > 0:
    weights = np.zeros((n_frames, n_morph), dtype=np.float32)
    for i in range(1, n_frames):
        weights[i, i - 1] = 1.0
    w_view = add_blob(weights.flatten().tobytes())
    w_acc  = add_accessor(w_view, pygltflib.FLOAT, n_frames * n_morph, pygltflib.SCALAR)

# ── ASSEMBLE GLTF ─────────────────────────────────────────────────────────────
binary = b''.join(blobs)
gltf   = pygltflib.GLTF2()
gltf.asset = pygltflib.Asset(version="2.0", generator="npz_to_animated_glb")

# Buffer
gltf.buffers = [pygltflib.Buffer(byteLength=len(binary))]

# Buffer views
for (offset, length, target) in views:
    bv = pygltflib.BufferView(buffer=0, byteOffset=offset, byteLength=length)
    if target is not None:
        bv.target = target
    gltf.bufferViews.append(bv)

# Accessors
for ad in accessors_meta:
    acc = pygltflib.Accessor(
        bufferView=ad["bufferView"],
        componentType=ad["componentType"],
        count=ad["count"],
        type=ad["type"],
    )
    if ad.get("min") is not None: acc.min = ad["min"]
    if ad.get("max") is not None: acc.max = ad["max"]
    gltf.accessors.append(acc)

# Material — warm skin tone
gltf.materials = [pygltflib.Material(
    name="skin",
    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
        baseColorFactor=[0.85, 0.72, 0.60, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.75,
    ),
    doubleSided=True,
)]

# Mesh with morph targets
targets = [{"POSITION": acc_i} for acc_i in morph_accs]
primitive = pygltflib.Primitive(
    attributes=pygltflib.Attributes(POSITION=pos_acc),
    indices=idx_acc,
    material=0,
    targets=targets,
)
initial_weights = [0.0] * n_morph
gltf.meshes = [pygltflib.Mesh(
    name="smplx_body", primitives=[primitive], weights=initial_weights
)]

# Node + Scene
gltf.nodes  = [pygltflib.Node(mesh=0, name="body")]
gltf.scenes = [pygltflib.Scene(nodes=[0])]
gltf.scene  = 0

# Animation
if n_morph > 0:
    sampler = pygltflib.AnimationSampler(
        input=t_acc, output=w_acc, interpolation="STEP"
    )
    channel = pygltflib.AnimationChannel(
        sampler=0,
        target=pygltflib.AnimationChannelTarget(node=0, path="weights"),
    )
    gltf.animations = [pygltflib.Animation(
        name="motion", samplers=[sampler], channels=[channel]
    )]

gltf.set_binary_blob(binary)
gltf.save_binary(OUTPUT_GLB)

size_kb = Path(OUTPUT_GLB).stat().st_size / 1024
print(f"[npz→glb] ✓ Saved → {OUTPUT_GLB}  ({size_kb:.0f} KB,  {n_frames} frames,  {n_morph} morph targets)")