"""Export intfloat/multilingual-e5-small to ONNX.

    python scripts/export_e5_onnx.py

Writes to `models/e5-small-onnx/` (gitignored — the graph alone is ~470 MB).

Why this exists
---------------
The agent loads e5-small through sentence-transformers, which brings PyTorch:
about 2 GB of library to run a model of 470 MB, for an inference that takes a
couple of hundred milliseconds per turn. On Lambda that library is not merely
large, it is *rented* — billing is memory x wall-clock, and the twenty seconds a
turn spends waiting on DeepSeek is twenty seconds of paying for 2 GB of PyTorch
doing nothing.

ONNX Runtime runs the same weights and only knows how to do inference, so the
runtime side needs onnxruntime + tokenizers + numpy and no torch at all.

This script is BUILD-TIME ONLY. `optimum` and `torch` are needed here and are
deliberately absent from the deployed image — see the multi-stage Dockerfile,
where the export happens in a stage that is thrown away.

Requires TWO packages, which is not obvious:

    pip install optimum optimum-onnx

Optimum 2.x moved the ONNX exporter into a separate distribution. Installing
`optimum[exporters]` alone gets you an `optimum-cli` whose `export onnx`
subcommand does not exist and a missing `optimum.exporters.onnx` — an error that
reads like a broken install rather than a missing package.

What must match, or recall degrades silently
--------------------------------------------
`modules.json` for this model declares three stages:

    0  Transformer
    1  Pooling      pooling_mode_mean_tokens: true
    2  Normalize

optimum exports stage 0 ONLY. Stages 1 and 2 are sentence-transformers' own
post-processing and have to be reimplemented on the runtime side — see
`shared/embedding.py::OnnxE5Backend`. Getting the mean wrong (dividing by the
padded length instead of the real token count) produces vectors that are still
384-dimensional and still searchable, just increasingly wrong as inputs get
longer. Nothing raises. `scripts/verify_onnx_parity.py` is what catches it.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "intfloat/multilingual-e5-small"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "models" / "e5-small-onnx"

# Files the runtime needs beyond model.onnx. tokenizer.json is the whole
# tokenizer (the Rust `tokenizers` library reads it directly, so `transformers`
# never has to be installed); the config files are what optimum writes alongside.
_REQUIRED = ("model.onnx", "tokenizer.json")


def _run_export(out_dir: Path) -> None:
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", MODEL_ID,
        "--task", "feature-extraction",
        str(out_dir),
    ]
    print(f"[export] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _verify(out_dir: Path) -> None:
    """Fail here rather than at Lambda INIT.

    A missing tokenizer.json surfaces in production as an exception naming a
    path, several layers below the reason, and only on the first request.
    """
    missing = [name for name in _REQUIRED if not (out_dir / name).exists()]
    if missing:
        raise SystemExit(
            f"[export] export finished but {missing} are missing from {out_dir}. "
            f"Present: {sorted(p.name for p in out_dir.iterdir())}"
        )

    size_mb = (out_dir / "model.onnx").stat().st_size / (1024 * 1024)
    total_mb = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / (1024 * 1024)
    print(f"[export] model.onnx {size_mb:.0f} MB · total {total_mb:.0f} MB")

    # The fp32 graph should land near the 470 MB of model.safetensors. An order
    # of magnitude below that means the export quantised or truncated something,
    # and the parity check would then fail for a reason that is not pooling.
    if size_mb < 200:
        raise SystemExit(
            f"[export] model.onnx is only {size_mb:.0f} MB; fp32 e5-small should be "
            f"~470 MB. Something was quantised or the export is incomplete."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--force", action="store_true",
                        help="re-export over an existing directory")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(f"[export] {args.out} already exists — use --force to re-export")
        _verify(args.out)
        return

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    _run_export(args.out)
    _verify(args.out)
    print(f"[export] done: {args.out}")


if __name__ == "__main__":
    main()
