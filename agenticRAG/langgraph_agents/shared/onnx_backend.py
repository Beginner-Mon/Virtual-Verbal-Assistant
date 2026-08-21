"""ONNX Runtime backend for e5-small — the three stages, by hand.

`shared/embedding.py` normally runs the model through sentence-transformers,
which brings PyTorch: ~2 GB of library to run a 470 MB model for a couple of
hundred milliseconds a turn. On Lambda that library is not just large, it is
*rented* — billing is memory x wall-clock, so the twenty seconds a turn spends
waiting on DeepSeek is twenty seconds of paying for PyTorch to do nothing. This
backend runs the same weights on onnxruntime, which only knows inference, so the
image drops from ~4 GB to ~1.2 GB and the function from 2048 MB to 1024 MB.

WHAT SENTENCE-TRANSFORMERS ACTUALLY DOES
----------------------------------------
`modules.json` for intfloat/multilingual-e5-small declares three stages, and
`optimum` exports only the first:

    0  Transformer                                  <- the .onnx file
    1  Pooling   pooling_mode_mean_tokens: true      <- reimplemented here
    2  Normalize                                     <- reimplemented here

Stages 1 and 2 are the whole risk of this backend. Both produce output of the
right shape whether or not they are right, so an error here does not raise, does
not log, and does not fail a test that only checks dimensions — it just returns
vectors that drift further from the reference the longer the input gets, and
recall falls quietly. `scripts/verify_onnx_parity.py` exists to catch exactly
that, by comparing against the torch path on real KB rows.

THE MEAN THAT IS EASY TO GET WRONG
----------------------------------
Mean pooling here is masked: sum the token vectors that are REAL and divide by
how many of those there were. Dividing by the padded sequence length instead is
the classic error — it silently scales every embedding down by the padding
ratio, which is harmless for a batch of one and increasingly wrong as batches
mix short and long texts. `max_seq_length` is 512 (sentence_bert_config.json),
and truncation has to match too, for the same reason.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# From sentence_bert_config.json. A different value here does not error — it
# just embeds a different amount of the text than the reference vectors did.
MAX_SEQ_LENGTH = 512

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "e5-small-onnx"


def model_dir() -> Path:
    """Where the exported model lives.

    E5_ONNX_DIR overrides, because the Lambda image bakes it somewhere else
    (`/opt/model`) than the repo checkout does.
    """
    return Path(os.getenv("E5_ONNX_DIR", str(_DEFAULT_MODEL_DIR)))


class OnnxE5Backend:
    """Tokenize -> ONNX -> masked mean -> L2 normalize.

    Loads lazily and behind a lock for the same reason the torch path does: the
    async wrappers call through asyncio.to_thread, so several parallel kb_search
    calls can reach a cold backend at once, and each would otherwise build its
    own session.
    """

    def __init__(self, directory: Optional[Path] = None):
        self._dir = Path(directory) if directory else model_dir()
        self._session = None
        self._tokenizer = None
        import threading
        self._load_lock = threading.Lock()

    # ── Loading ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._load_lock:
            if self._session is not None:
                return

            import onnxruntime as ort
            from tokenizers import Tokenizer

            model_path = self._dir / "model.onnx"
            tokenizer_path = self._dir / "tokenizer.json"
            if not model_path.exists() or not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found in {self._dir}. Build it with:\n"
                    f"    python scripts/export_e5_onnx.py\n"
                    f"On Lambda this means the image was built without the export "
                    f"stage, or E5_ONNX_DIR points somewhere else."
                )

            # `tokenizers`, not `transformers`: the Rust library reads
            # tokenizer.json directly and is ~3 MB, where transformers would drag
            # in most of what this backend exists to avoid.
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
            tokenizer.enable_padding()
            self._tokenizer = tokenizer

            # One thread each. Lambda gives ~0.58 vCPU at 1024 MB, so letting
            # onnxruntime spawn a pool per session buys contention, not speed —
            # and the async wrappers already run this off the event loop.
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(os.getenv("ONNX_INTRA_OP_THREADS", "1"))
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

    # ── Inference ────────────────────────────────────────────────────────

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed already-prefixed texts. Returns (n, 384) float32, L2-normalized.

        Prefixing is the caller's job — E5EmbeddingService owns `query: ` and
        `passage: ` so that both backends cannot disagree about them.
        """
        self._ensure_loaded()

        encodings = self._tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        # XLM-R has no token_type_ids, but the exported graph may still declare
        # the input. Feed zeros rather than assume, because a missing required
        # input fails at run time with a message about names, not about models.
        declared = {i.name for i in self._session.get_inputs()}
        if "token_type_ids" in declared:
            feed["token_type_ids"] = np.zeros_like(input_ids)

        token_vectors = self._session.run(None, feed)[0]      # (n, seq, 384)

        return _l2_normalize(_masked_mean(token_vectors, attention_mask))


def _masked_mean(token_vectors: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Stage 1 — Pooling(pooling_mode_mean_tokens=True).

    Sum the REAL tokens and divide by how many there were. The denominator is
    the mask sum, never `token_vectors.shape[1]`: padding is not content, and
    dividing by the padded length scales every vector down by the padding ratio.
    That is invisible with one text per call and grows with batch heterogeneity —
    which is why the parity check batches long and short inputs together.
    """
    mask = attention_mask.astype(np.float32)[..., None]       # (n, seq, 1)
    summed = (token_vectors * mask).sum(axis=1)               # (n, 384)
    # clip: an all-padding row would divide by zero and yield NaN, which
    # propagates into pgvector as a row that matches nothing and explains itself
    # to nobody.
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Stage 2 — Normalize. Same epsilon guard, same reason."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, a_min=1e-12, a_max=None)).astype(np.float32)
