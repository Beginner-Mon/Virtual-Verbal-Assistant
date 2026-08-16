#!/usr/bin/env python3
"""Upload VRM models to S3 and seed the `characters` table.

For each model in ECA_UI/frontend/src/asset/models/*.vrm:

    1. parse the GLB for humanoid + blendShape metadata
    2. upload to s3://<bucket>/characters/{slug}/{sha256[:8]}.vrm
    3. read personas/{slug}.md via the backend's own parser
    4. read the avatar profile via the frontend's own module graph
    5. UPSERT one row into characters

Re-runnable: keyed on slug, and an unchanged file produces the same content
hash and therefore the same S3 key.

    python scripts/upload_characters_to_s3.py --dry-run    # no AWS, no DB
    python scripts/upload_characters_to_s3.py --bucket vva-assets-123456789012 \\
        --cdn https://d111111abcdef8.cloudfront.net

--dry-run needs nothing but Node and the repo, so metadata extraction can be
checked against ECA_UI/frontend/src/avatar/vrmManifest.ts before any
credentials exist.

Requires VVA_PG_DSN for the real run — the same variable the backend and
Alembic read, so there is no way to seed one database while the app reads
another.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "ECA_UI" / "frontend" / "src" / "asset" / "models"
PROFILE_EXPORTER = REPO_ROOT / "ECA_UI" / "frontend" / "scripts" / "export-avatar-profiles.mjs"

sys.path.insert(0, str(REPO_ROOT / "agenticRAG"))

# ── VRM 0.x preset categories ───────────────────────────────────────
# Kept identical to ECA_UI/frontend/scripts/extract-vrm-meta.mjs so the numbers
# this script writes to the DB match the manifest the frontend already ships.

EMOTION_PRESETS = {"joy", "angry", "sorrow", "fun", "neutral"}
VISEME_PRESETS = {"a", "i", "u", "e", "o"}
BLINK_PRESETS = {"blink", "blink_l", "blink_r"}
LOOKAT_PRESETS = {"lookup", "lookdown", "lookleft", "lookright"}

# Torso chain used for the retarget-compatibility check. Motion retargeting
# distributes rotation across these; a rig with fewer than three of them cannot
# reproduce a spine curve and produces visibly broken playback.
SPINE_CHAIN = ("hips", "spine", "chest", "upperChest")

GLB_MAGIC = 0x46546C67  # "glTF"


def classify(preset_name: str | None) -> str:
    if not preset_name or preset_name == "unknown":
        return "custom"
    if preset_name in EMOTION_PRESETS:
        return "emotion"
    if preset_name in VISEME_PRESETS:
        return "viseme"
    if preset_name in BLINK_PRESETS:
        return "blink"
    if preset_name in LOOKAT_PRESETS:
        return "lookAt"
    return "custom"


def read_glb_json(path: Path) -> dict:
    """Return the JSON chunk of a GLB file.

    Same layout the frontend's extract-vrm-meta.mjs walks: a 12-byte header,
    then chunk length at offset 12 and the JSON payload from offset 20.
    """
    data = path.read_bytes()
    if len(data) < 20:
        raise ValueError("file too short to be a GLB")
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB file (bad magic)")
    json_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:  # "JSON"
        raise ValueError("first GLB chunk is not JSON")
    return json.loads(data[20:20 + json_len].decode("utf-8"))


def extract_vrm_metadata(path: Path) -> dict:
    """Build the characters.vrm_metadata payload for one .vrm file."""
    glb = read_glb_json(path)
    extensions = glb.get("extensions") or {}

    # VRM 0.x lives under "VRM"; VRM 1.0 under "VRMC_vrm". All four current
    # models are 0.x, but reading both keeps a 1.0 model from silently
    # extracting as an empty rig.
    vrm0 = extensions.get("VRM")
    vrm1 = extensions.get("VRMC_vrm")
    ext = vrm0 or vrm1 or {}
    spec_version = ext.get("specVersion") or ("0.0" if vrm0 else "1.0" if vrm1 else "unknown")

    # Joint count: the real skeleton size, i.e. the union of every skin's joint
    # list — not the humanoid bone map, which only covers standard bones and
    # would undercount every model with hair or skirt bones.
    joints: set[int] = set()
    for skin in glb.get("skins") or []:
        joints.update(skin.get("joints") or [])

    if vrm0:
        human_bones = (ext.get("humanoid") or {}).get("humanBones") or []
        bone_names = {b.get("bone") for b in human_bones if b.get("bone")}
        groups = (ext.get("blendShapeMaster") or {}).get("blendShapeGroups") or []
    else:
        human_bones = (ext.get("humanoid") or {}).get("humanBones") or {}
        bone_names = set(human_bones.keys()) if isinstance(human_bones, dict) else set()
        groups = []  # VRM 1.0 uses expressions, not blendShapeGroups

    counts = {"emotions": 0, "visemes": 0, "blinks": 0, "look_ats": 0, "customs": 0}
    key = {
        "emotion": "emotions", "viseme": "visemes",
        "blink": "blinks", "lookAt": "look_ats", "custom": "customs",
    }
    for g in groups:
        counts[key[classify(g.get("presetName"))]] += 1

    spine_count = sum(1 for b in SPINE_CHAIN if b in bone_names)

    reasons: list[str] = []
    if spine_count < 3:
        reasons.append("spine_count < 3")
    if sum(counts.values()) == 0:
        reasons.append("blendshape_groups.total == 0")
    if not bone_names:
        reasons.append("no humanoid rig")

    return {
        "joint_count": len(joints),
        "spine_count": spine_count,
        "has_humanoid_rig": bool(bone_names),
        "blendshape_groups": {"total": sum(counts.values()), **counts},
        "has_blink": counts["blinks"] > 0,
        "has_look_at": counts["look_ats"] > 0,
        "incompatible_reasons": reasons,
        "vrm_version": spec_version,
        "file_size_bytes": path.stat().st_size,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }


def load_avatar_profiles(slugs: list[str]) -> dict:
    """Resolve each slug's avatar profile by running the frontend's own loader.

    Shelling out to Node rather than parsing the .ts: bronya.ts spreads
    defaultProfile, so any file-at-a-time parser yields a profile with no
    recipes and no visemes, and the avatar renders expressionless.
    """
    proc = subprocess.run(
        ["node", str(PROFILE_EXPORTER), *slugs],
        capture_output=True, text=True, encoding="utf-8",
        cwd=str(PROFILE_EXPORTER.parent.parent),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"export-avatar-profiles.mjs failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def load_persona(slug: str) -> dict:
    """Parse personas/{slug}.md with the backend's own loader.

    Importing _load_persona rather than reimplementing it means the JSONB in
    the DB cannot drift from what get_persona() produces at runtime.
    """
    from langgraph_agents.nodes._persona_loader import _load_persona

    persona = _load_persona(slug)
    if persona.get("_fallback"):
        raise FileNotFoundError(
            f"personas/{slug}.md is missing or unparseable — refusing to seed a "
            f"fallback persona for character '{slug}'"
        )
    return persona


def display_name_for(slug: str, persona: dict) -> str:
    """Prefer the persona's declared Name, fall back to a title-cased slug."""
    identity = persona.get("identity", "")
    for field in identity.split("|"):
        label, _, value = field.partition(":")
        if label.strip().lower() == "name" and value.strip():
            return value.strip()
    return slug.replace("-", " ").replace("_", " ").title()


def build_records(cdn_base: str) -> list[dict]:
    vrm_files = sorted(MODELS_DIR.glob("*.vrm"))
    if not vrm_files:
        raise SystemExit(f"No .vrm files found in {MODELS_DIR}")

    slugs = [f.stem for f in vrm_files]
    profiles = load_avatar_profiles(slugs)

    records = []
    for order, path in enumerate(vrm_files):
        slug = path.stem
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
        # `models/`, not `characters/`: CloudFront routes /characters* to the
        # catalog Lambda, so an object at characters/anne/<hash>.vrm would be
        # served by the API instead of S3 — the Lambda sees a third path segment
        # that is not "avatar-profile" and answers 404. Keeping the object prefix
        # out of the API's namespace is what stops the two colliding.
        key = f"models/{slug}/{digest}.vrm"
        persona = load_persona(slug)

        records.append({
            "slug": slug,
            "display_name": display_name_for(slug, persona),
            "description": None,
            "local_path": path,
            "s3_key": key,
            "vrm_url": f"{cdn_base.rstrip('/')}/{key}" if cdn_base else key,
            "vrm_metadata": extract_vrm_metadata(path),
            "avatar_profile": profiles.get(slug, {}),
            "persona": persona,
            "voice_language": persona.get("voice_identity", {}).get("language", "vi"),
            "sort_order": order,
        })
    return records


def upload(records: list[dict], bucket: str) -> None:
    import boto3

    s3 = boto3.client("s3")
    for rec in records:
        print(f"  uploading {rec['local_path'].name} -> s3://{bucket}/{rec['s3_key']}")
        s3.upload_file(
            str(rec["local_path"]), bucket, rec["s3_key"],
            ExtraArgs={
                "ContentType": "model/gltf-binary",
                # The key is content-addressed, so the object at this URL can
                # never change. Cache it for a year.
                "CacheControl": "public, max-age=31536000, immutable",
            },
        )


async def upsert(records: list[dict]) -> None:
    from langgraph_agents.shared import get_pg_client

    pg = get_pg_client()
    await pg.connect()
    for rec in records:
        await pg.execute(
            """
            INSERT INTO characters (
                slug, display_name, description, vrm_url, vrm_metadata,
                avatar_profile, persona, voice_language, sort_order
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8, $9)
            ON CONFLICT (slug) DO UPDATE SET
                display_name   = EXCLUDED.display_name,
                vrm_url        = EXCLUDED.vrm_url,
                vrm_metadata   = EXCLUDED.vrm_metadata,
                avatar_profile = EXCLUDED.avatar_profile,
                persona        = EXCLUDED.persona,
                voice_language = EXCLUDED.voice_language,
                sort_order     = EXCLUDED.sort_order,
                updated_at     = now()
            """,
            rec["slug"], rec["display_name"], rec["description"], rec["vrm_url"],
            json.dumps(rec["vrm_metadata"]), json.dumps(rec["avatar_profile"]),
            json.dumps(rec["persona"]), rec["voice_language"], rec["sort_order"],
        )
        print(f"  upserted {rec['slug']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", help="S3 bucket (AssetStack output AssetBucketName)")
    ap.add_argument("--cdn", default="", help="CloudFront base URL (AssetStack output AssetBaseUrl)")
    ap.add_argument("--dry-run", action="store_true", help="extract and print only; no AWS, no DB")
    args = ap.parse_args()

    if not args.dry_run and not (args.bucket and args.cdn):
        ap.error("--bucket and --cdn are required unless --dry-run")

    records = build_records(args.cdn)

    print(f"\n{len(records)} character(s) from {MODELS_DIR}\n")
    for rec in records:
        m = rec["vrm_metadata"]
        bs = m["blendshape_groups"]
        warn = " ".join(f"[{r}]" for r in m["incompatible_reasons"]) or "compatible"
        print(f"  {rec['slug']:14s} {rec['display_name']:12s} "
              f"{m['file_size_bytes'] / 1e6:5.1f}MB  vrm={m['vrm_version']}  "
              f"joints={m['joint_count']:3d} spine={m['spine_count']}  "
              f"bs={bs['total']:2d} (E{bs['emotions']} V{bs['visemes']} B{bs['blinks']} "
              f"L{bs['look_ats']} C{bs['customs']})  {warn}")
        print(f"  {'':14s} {rec['vrm_url']}")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded, nothing written.")
        return

    print(f"\nUploading to s3://{args.bucket} ...")
    upload(records, args.bucket)

    print("\nSeeding characters ...")
    asyncio.run(upsert(records))
    print("\nDone.")


if __name__ == "__main__":
    main()
