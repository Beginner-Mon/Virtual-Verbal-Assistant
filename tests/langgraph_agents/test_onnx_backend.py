"""The two stages sentence-transformers does after the model, tested as maths.

`shared/onnx_backend.py` reimplements Pooling(mean) and Normalize, because
`optimum` exports only the Transformer stage. Both reimplementations return
arrays of the right shape whether or not they are correct, so nothing about
their output raises, logs, or fails a shape assertion when they are wrong — the
vectors are simply a little off, more so for longer and more heterogeneous
inputs, and retrieval quality drops without a single error.

`scripts/verify_onnx_parity.py` is the real check: it compares against the torch
path on actual KB rows. It cannot run in CI (needs the 465 MB export and a
database), so these tests pin the arithmetic that parity run validated —
specifically the denominators, which is where the plausible bug lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from langgraph_agents.shared.onnx_backend import (
    MAX_SEQ_LENGTH,
    _l2_normalize,
    _masked_mean,
)


# ── Masked mean: the denominator is the whole risk ────────────────────────────


@pytest.mark.unit
def test_mean_ignores_padding():
    """THE test. Padding must not enter the average.

    Two real tokens ([2,2,2] and [4,4,4]) then two padded positions carrying
    garbage. The mean of the real tokens is 3. Dividing by the padded length of
    4 instead would give 1.5 — a vector scaled by the padding ratio, which is
    still 3-dimensional, still normalizable, and still wrong.
    """
    tokens = np.array([[[2.0] * 3, [4.0] * 3, [99.0] * 3, [99.0] * 3]])
    mask = np.array([[1, 1, 0, 0]])

    result = _masked_mean(tokens, mask)

    np.testing.assert_allclose(result, [[3.0, 3.0, 3.0]])


@pytest.mark.unit
def test_mixed_lengths_in_one_batch_are_each_divided_by_their_own_count():
    """Where a padding-ratio bug actually shows up.

    One text per call hides it — the padded length equals the real length when
    there is nothing to pad against. In a batch the short row is padded to the
    long row's width, so each row needs its OWN denominator. This is why
    verify_onnx_parity.py sorts its sample by length before batching.
    """
    tokens = np.array([
        [[6.0], [0.0], [0.0]],        # 1 real token  → 6.0
        [[2.0], [4.0], [6.0]],        # 3 real tokens → 4.0
    ])
    mask = np.array([[1, 0, 0], [1, 1, 1]])

    result = _masked_mean(tokens, mask)

    np.testing.assert_allclose(result, [[6.0], [4.0]])


@pytest.mark.unit
def test_all_padding_row_does_not_produce_nan():
    """A degenerate row must not poison the batch.

    A NaN here reaches pgvector as a row that matches nothing and explains
    itself to nobody — worse than an empty result, because it looks like data.
    """
    tokens = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    mask = np.array([[0, 0]])

    result = _masked_mean(tokens, mask)

    assert np.isfinite(result).all(), f"non-finite output: {result}"


# ── L2 normalize ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_normalize_gives_unit_vectors():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [-5.0, 12.0]])

    norms = np.linalg.norm(_l2_normalize(vectors), axis=1)

    np.testing.assert_allclose(norms, [1.0, 1.0, 1.0], rtol=1e-6)


@pytest.mark.unit
def test_normalize_survives_a_zero_vector():
    """Follows from the all-padding case above: mean 0 then normalize."""
    result = _l2_normalize(np.array([[0.0, 0.0, 0.0]]))

    assert np.isfinite(result).all()


@pytest.mark.unit
def test_normalize_returns_float32():
    """pgvector stores float4; returning float64 would cost a conversion per
    query and hide the precision the parity check actually measured."""
    assert _l2_normalize(np.array([[3.0, 4.0]])).dtype == np.float32


# ── Configuration that must match the reference ───────────────────────────────


@pytest.mark.unit
def test_max_seq_length_matches_the_model_card():
    """512, from sentence_bert_config.json.

    Not a magic number to tidy away: a different value truncates a different
    amount of each document than the torch path did when the 2918 stored vectors
    were produced, and the two would then be embedding different texts.
    """
    assert MAX_SEQ_LENGTH == 512


@pytest.mark.unit
def test_backend_selection_rejects_a_typo():
    """A bad EMBEDDING_BACKEND must fail loudly, not fall back.

    Falling back to torch would mean an image built for ONNX silently loading a
    PyTorch that is not in it — an ImportError at the first retrieval instead of
    a clear error at construction.
    """
    from langgraph_agents.shared.embedding import E5EmbeddingService

    with pytest.raises(ValueError, match="onnxruntime|not a backend"):
        E5EmbeddingService(backend="onnxruntime")


# ── Preflight must not cry wolf about the backend it is not using ────────────


@pytest.mark.unit
def test_preflight_skips_the_backend_that_is_not_in_use(monkeypatch):
    """The ONNX image ships neither torch nor sentence-transformers on purpose.

    Without the `required_when` predicate, preflight would log
    "MISSING CRITICAL: sentence_transformers" on every cold start of a perfectly
    healthy deployment — and a critical alarm that is always wrong is how people
    learn to scroll past the real ones.
    """
    from langgraph_agents.shared import preflight

    monkeypatch.setenv("EMBEDDING_BACKEND", "onnx")
    required = {d.module for d in preflight.LAZY_DEPENDENCIES if d.is_required()}
    assert "sentence_transformers" not in required
    assert {"onnxruntime", "tokenizers"} <= required

    monkeypatch.setenv("EMBEDDING_BACKEND", "torch")
    required = {d.module for d in preflight.LAZY_DEPENDENCIES if d.is_required()}
    assert "sentence_transformers" in required
    assert not ({"onnxruntime", "tokenizers"} & required)


@pytest.mark.unit
def test_preflight_predicate_is_evaluated_late(monkeypatch):
    """Read at call time, not import time.

    LAZY_DEPENDENCIES is a module-level tuple built once. If the predicate had
    captured the environment when that tuple was constructed, a process
    configured after import — which is every Lambda, since the runtime injects
    variables before the handler but after the module graph loads — would check
    the wrong pair.
    """
    from langgraph_agents.shared import preflight

    dep = next(d for d in preflight.LAZY_DEPENDENCIES
               if d.module == "sentence_transformers")

    monkeypatch.setenv("EMBEDDING_BACKEND", "onnx")
    assert dep.is_required() is False
    monkeypatch.setenv("EMBEDDING_BACKEND", "torch")
    assert dep.is_required() is True


@pytest.mark.unit
def test_default_backend_is_torch():
    """The 2918 vectors in pgvector came from the torch path.

    The default stays torch even though parity passed, because the ONNX export
    is a 465 MB gitignored build artifact: defaulting to it would break every
    checkout that has not run scripts/export_e5_onnx.py. The Lambda image sets
    EMBEDDING_BACKEND=onnx explicitly instead.
    """
    import os
    from langgraph_agents.shared.embedding import E5EmbeddingService

    previous = os.environ.pop("EMBEDDING_BACKEND", None)
    try:
        assert E5EmbeddingService().backend == "torch"
    finally:
        if previous is not None:
            os.environ["EMBEDDING_BACKEND"] = previous
