"""Build the streaming-probe zip. No Docker.

    python infra/spike/build_probe.py

Same two tricks as infra/build_crud_api.py, for the same two reasons:

  * run.sh must be mode 0755 inside the archive, and Windows has no such bit —
    so the archive is written here with the mode set explicitly rather than
    letting CDK zip whatever the filesystem reports.
  * dependencies are downloaded as manylinux wheels, so a Windows machine with
    no Docker produces a package that loads on Amazon Linux.

Only fastapi + uvicorn: the probe answers a question about the transport, so
anything else in the package is a second variable.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_SPIKE_ROOT = Path(__file__).resolve().parent
_SRC = _SPIKE_ROOT / "streaming_probe"
_BUILD = _SPIKE_ROOT.parent / "build"
_STAGE = _BUILD / "streaming_probe"
_ZIP = _BUILD / "streaming_probe.zip"

_PYTHON_VERSION = "3.12"
_PLATFORM = "manylinux2014_x86_64"
_REQUIREMENTS = ("fastapi>=0.130,<0.140", "uvicorn>=0.30,<1.0", "pydantic>=2")


def main() -> None:
    if _STAGE.exists():
        shutil.rmtree(_STAGE)
    _STAGE.mkdir(parents=True)

    print(f"[probe] installing for {_PLATFORM} / py{_PYTHON_VERSION}")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", *_REQUIREMENTS,
            "--target", str(_STAGE),
            "--platform", _PLATFORM,
            "--only-binary=:all:",
            "--python-version", _PYTHON_VERSION,
            "--implementation", "cp",
            "--no-cache-dir", "--disable-pip-version-check", "--quiet",
        ],
        check=True,
    )

    shutil.copy(_SRC / "app.py", _STAGE / "app.py")
    # Read as text and rewrite the newlines: CRLF would make the shebang
    # `#!/bin/bash\r`, and the kernel then looks for an interpreter whose name
    # ends in a carriage return. The error says "no such file or directory"
    # about a path that plainly exists.
    (_STAGE / "run.sh").write_bytes(
        (_SRC / "run.sh").read_text(encoding="utf-8").replace("\r\n", "\n").encode()
    )

    strays = [p for p in _STAGE.rglob("*") if p.suffix.lower() in (".pyd", ".dll")]
    if strays:
        raise SystemExit(
            f"[probe] {len(strays)} Windows binaries in the package — pip did not "
            f"use manylinux wheels: {sorted({p.name for p in strays})[:5]}"
        )

    _ZIP.parent.mkdir(parents=True, exist_ok=True)
    if _ZIP.exists():
        _ZIP.unlink()
    with zipfile.ZipFile(_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(_STAGE.rglob("*")):
            if path.is_dir():
                continue
            arcname = path.relative_to(_STAGE).as_posix()
            info = zipfile.ZipInfo(arcname)
            info.external_attr = (0o755 if arcname == "run.sh" else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())

    shutil.rmtree(_STAGE)
    print(f"[probe] {_ZIP} — {_ZIP.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
