"""Build-time check: the exported ONNX model loads and runs. Run inside the image.

    python agenticRAG/verify_onnx_export.py /out

`test -f model.onnx` proves a file arrived. This proves it is a model — the graph
loads, accepts the inputs shared/onnx_backend.py feeds it, and returns the shape
that backend expects.

Worth a build step because the alternative is discovering it at Lambda INIT: in
CloudWatch, minutes after the fact, on a ten-minute loop, with a message about
onnxruntime rather than about the export.

A separate file rather than `RUN python -c "..."` for one reason: a multi-line
`python -c` inside a Dockerfile is a shell-quoting exercise where a mistake costs
the same ten-minute loop this check exists to avoid. This one can be run on a
laptop against the same export.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Must match shared/onnx_backend.py. A mismatch here would let a model through
# that the runtime then drives differently from how it was checked.
MAX_SEQ_LENGTH = 512
EXPECTED_DIM = 384


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "/out")

    model_path = directory / "model.onnx"
    tokenizer_path = directory / "tokenizer.json"
    for path in (model_path, tokenizer_path):
        if not path.exists():
            print(f"[verify] missing {path}")
            return 1

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
    tokenizer.enable_padding()

    # Two texts of different lengths, deliberately: padding only exists in a
    # batch, and a single-text check would not exercise the attention mask the
    # runtime relies on for its masked mean.
    encodings = tokenizer.encode_batch(["query: smoke test", "passage: a rather longer one"])
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    declared = {i.name for i in session.get_inputs()}
    if "token_type_ids" in declared:
        feed["token_type_ids"] = np.zeros_like(input_ids)

    missing = declared - set(feed)
    if missing:
        print(f"[verify] graph wants inputs the runtime does not feed: {sorted(missing)}")
        return 1

    output = session.run(None, feed)[0]

    # 3-D means last_hidden_state — the transformer body, pooling NOT included.
    # If a future export ever emits a pooled (n, 384) instead, onnx_backend's
    # _masked_mean would broadcast against the wrong axis, and the failure would
    # be a shape error at the first search rather than here.
    if output.ndim != 3 or output.shape[-1] != EXPECTED_DIM:
        print(
            f"[verify] expected (batch, seq, {EXPECTED_DIM}) last_hidden_state, "
            f"got {output.shape}. If this is (batch, {EXPECTED_DIM}) the export "
            f"included pooling, and shared/onnx_backend.py must stop applying its "
            f"own."
        )
        return 1

    print(f"[verify] OK — inputs {sorted(declared)}, output {output.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
