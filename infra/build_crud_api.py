"""Build the deployment zip for the CRUD API Lambda.

    python infra/build_crud_api.py

Produces infra/build/crud_api.zip, which infra/infra/crud_api_stack.py deploys.

Why a build script instead of CDK's own directory bundling
----------------------------------------------------------
Two reasons, both about running this from Windows.

1. **Executable bits.** The AWS Lambda Web Adapter starts the function by
   exec'ing `run.sh`, so that file must be mode 0755 inside the zip. Windows has
   no such bit, and CDK's directory zipping writes whatever the filesystem
   reports — the function then fails at INIT with "permission denied", which
   looks nothing like the cause. Writing the archive here means setting the mode
   explicitly.

2. **Linux wheels without Docker.** asyncpg, pydantic-core and cryptography ship
   compiled binaries; a wheel built on Windows produces `.pyd` files that will
   not load on Amazon Linux. CDK's answer is to bundle inside a container, which
   needs Docker running. `pip --platform manylinux2014_x86_64 --only-binary=:all:`
   downloads the prebuilt Linux wheels straight from PyPI instead, so the build
   works on a machine with no Docker at all. The same escape hatch is described
   in infra/infra/character_stack.py.

The zip contains the whole `langgraph_agents` package, not just the CRUD
modules. Source is small; what costs size is dependencies, and those are pinned
by lambda/crud_api/requirements.txt. Shipping the package whole keeps one copy
of the code — the alternative is a trimmed parallel tree that drifts, which is
exactly what the hand-written handlers under infra/lambda/ did.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_INFRA_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _INFRA_ROOT.parent
_SRC_PACKAGE = _REPO_ROOT / "agenticRAG" / "langgraph_agents"
_LAMBDA_DIR = _INFRA_ROOT / "lambda" / "crud_api"
_BUILD_DIR = _INFRA_ROOT / "build"
_STAGE_DIR = _BUILD_DIR / "crud_api"
_ZIP_PATH = _BUILD_DIR / "crud_api.zip"

# Matches the Lambda runtime in crud_api_stack.py. Both must move together.
_PYTHON_VERSION = "3.12"
_PLATFORM = "manylinux2014_x86_64"

# Kept out of the archive: caches, local logs, and the migration tooling, which
# runs from a developer machine against the database and has no business in a
# function that only reads and writes rows.
_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", "*.log", ".pytest_cache", "alembic",
)


def _install_dependencies(target: Path) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "-r", str(_LAMBDA_DIR / "requirements.txt"),
            "--target", str(target),
            "--platform", _PLATFORM,
            "--only-binary=:all:",
            "--python-version", _PYTHON_VERSION,
            "--implementation", "cp",
            "--no-cache-dir",
            "--disable-pip-version-check",
        ],
        check=True,
    )


def _assert_no_windows_binaries(staged: Path) -> None:
    """Fail the build if a Windows extension slipped in.

    The whole point of --platform/--only-binary is that every compiled
    dependency arrives as a Linux `.so`. A stray `.pyd` means pip fell back to
    something host-specific, and the failure would otherwise surface as an
    ImportError at Lambda INIT — after a deploy, with a stack trace that names
    the module and not the reason.
    """
    strays = [p for p in staged.rglob("*") if p.suffix.lower() in (".pyd", ".dll")]
    if strays:
        names = ", ".join(sorted({p.name for p in strays})[:5])
        raise SystemExit(
            f"[build] {len(strays)} Windows binary file(s) in the staged package "
            f"({names}). pip did not use manylinux wheels — check that every "
            f"requirement publishes one for {_PLATFORM}/cp{_PYTHON_VERSION}."
        )


def _write_zip(staged: Path, zip_path: Path) -> None:
    """Zip the staged tree, giving run.sh the executable bit."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(staged.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(staged).as_posix()
            info = zipfile.ZipInfo(arcname)
            # 0755 for the entry point LWA execs, 0644 for everything else.
            # zipfile stores Unix modes in the top 16 bits of external_attr.
            mode = 0o755 if arcname == "run.sh" else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-stage", action="store_true",
        help="leave build/crud_api/ in place for inspection",
    )
    args = parser.parse_args()

    if _STAGE_DIR.exists():
        shutil.rmtree(_STAGE_DIR)
    _STAGE_DIR.mkdir(parents=True)

    print(f"[build] installing dependencies for {_PLATFORM} / py{_PYTHON_VERSION}")
    _install_dependencies(_STAGE_DIR)

    print("[build] staging langgraph_agents")
    shutil.copytree(
        _SRC_PACKAGE, _STAGE_DIR / "langgraph_agents",
        ignore=_IGNORE, dirs_exist_ok=True,
    )

    # run.sh is copied last so it is unmistakably the entry point, and read back
    # in text mode below to normalise line endings.
    run_sh = (_LAMBDA_DIR / "run.sh").read_text(encoding="utf-8")
    # CRLF would make the shebang line `#!/bin/bash\r`, and the kernel would look
    # for an interpreter whose name ends in a carriage return. The error is
    # "no such file or directory" naming a path that plainly exists.
    (_STAGE_DIR / "run.sh").write_bytes(run_sh.replace("\r\n", "\n").encode("utf-8"))

    _assert_no_windows_binaries(_STAGE_DIR)

    print(f"[build] writing {_ZIP_PATH}")
    _write_zip(_STAGE_DIR, _ZIP_PATH)

    size_mb = _ZIP_PATH.stat().st_size / (1024 * 1024)
    unzipped_mb = sum(p.stat().st_size for p in _STAGE_DIR.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"[build] done: {size_mb:.1f} MB zipped, {unzipped_mb:.1f} MB unzipped")
    if unzipped_mb > 250:
        raise SystemExit(
            f"[build] {unzipped_mb:.0f} MB unzipped exceeds Lambda's 250 MB limit "
            f"for zip packages. Something heavy got in — check requirements.txt."
        )

    if not args.keep_stage:
        shutil.rmtree(_STAGE_DIR)


if __name__ == "__main__":
    main()
