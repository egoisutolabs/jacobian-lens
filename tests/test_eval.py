# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""run_eval end-to-end on the tiny CPU model, plus readout-position logic."""

from __future__ import annotations

import pytest
import torch

from jlens.eval import _readout_position, _single_token_ids, load_eval, run_eval
from jlens.fitting import fit

from .tiny import TinyDecoder


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


@pytest.fixture(scope="module")
def lens(model):
    return fit(model, ["abcdefghij " * 5], source_layers=[0, 1, 2], dim_batch=4)


# Single-char intermediates so the byte tokenizer yields single-token forms.
EVAL = {
    "name": "toy",
    "items": [
        {"prompt": "the cat sat on the mat", "intermediates": ["a", "t"]},
        {"prompt": "a dog ran far away today", "intermediates": ["o"]},
    ],
}


def test_pass_at_k_bounded_and_monotonic(model, lens):
    result = run_eval(model, lens, EVAL, ks=(1, 5, 10))
    assert result.n_items == 2
    vals = [result.pass_at_k[k] for k in (1, 5, 10)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert vals[0] <= vals[1] <= vals[2]  # pass@k is non-decreasing in k


def test_pass_at_full_vocab_is_one(model, lens):
    # Every intermediate here has a single-token form, so at k == vocab_size
    # every rank (< vocab) counts as a hit and pass@k must be exactly 1.0.
    vocab = model.lm_head.weight.shape[0]
    result = run_eval(model, lens, EVAL, ks=(vocab,))
    assert result.pass_at_k[vocab] == pytest.approx(1.0)


def test_unencodable_intermediate_is_a_miss(model, lens):
    # A multi-char word is multi-token under the byte tokenizer -> no single
    # token form -> None -> never a hit, even at k == vocab_size.
    vocab = model.lm_head.weight.shape[0]
    result = run_eval(
        model, lens, {"items": [{"prompt": "hello", "intermediates": ["word"]}]},
        ks=(vocab,),
    )
    assert result.best_ranks[0]["word"] is None
    assert result.pass_at_k[vocab] == 0.0


def test_single_token_ids_filters_multi_token(model):
    tok = model.tokenizer
    assert _single_token_ids(tok, "a")  # single char -> at least one form
    assert _single_token_ids(tok, "elephant") == []  # multi-token everywhere


def test_readout_position_last_token_by_default():
    class _Tok:
        def decode(self, ids, **_):
            return "x"

    ids = torch.tensor([[5, 6, 7, 8]])
    assert _readout_position("lens-eval-typo", ids, _Tok()) == 3


def test_readout_position_poetry_is_last_newline():
    class _Tok:
        def decode(self, ids, **_):
            return "\n" if ids[0] == 9 else "x"

    ids = torch.tensor([[1, 9, 2, 9, 3]])  # newlines at positions 1 and 3
    assert _readout_position("lens-eval-poetry", ids, _Tok()) == 3


def test_load_eval_from_slug_if_present():
    # Repo checkout ships the real sets; skip cleanly if run from a wheel.
    try:
        name, items = load_eval("lens-eval-typo")
    except FileNotFoundError:
        pytest.skip("eval data not present (installed, not a source checkout)")
    assert name == "lens-eval-typo"
    assert items and "intermediates" in items[0]
