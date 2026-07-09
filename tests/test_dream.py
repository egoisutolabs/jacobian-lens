# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""dream/consolidate/remember end-to-end on the tiny CPU model."""

from __future__ import annotations

import json

import pytest
import torch

from jlens.dream import (
    Concept,
    DreamStep,
    _resolve_layers,
    _sample,
    consolidate,
    dream,
    remember,
)
from jlens.fitting import fit

from .tiny import TinyDecoder


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


@pytest.fixture(scope="module")
def lens(model):
    # source_layers must be < target (final) layer 3.
    return fit(model, ["abcdefghij " * 5], source_layers=[0, 1, 2], skip_first=1)


SEED = "I keep thinking about the "


def test_dream_shape(model, lens):
    result = dream(model, lens, SEED, n_steps=12, layers=[1, 2], temperature=0.0)
    assert len(result.steps) == 12
    assert result.layers == [1, 2]
    for step in result.steps:
        assert isinstance(step, DreamStep)
        assert set(step.latent) == {1, 2}
        assert all(len(ids) > 0 for ids in step.latent.values())
    # surface_text is the concatenation of the emitted tokens.
    assert result.surface_text == model.tokenizer.decode(
        [s.surface_id for s in result.steps]
    )


def test_greedy_is_deterministic(model, lens):
    a = dream(model, lens, SEED, n_steps=16, layers=[1, 2], temperature=0.0)
    b = dream(model, lens, SEED, n_steps=16, layers=[1, 2], temperature=0.0)
    assert a.surface_text == b.surface_text
    assert [c.token_id for c in a.concepts] == [c.token_id for c in b.concepts]


def test_seeded_sampling_is_reproducible(model, lens):
    kw = dict(n_steps=16, layers=[1, 2], temperature=0.9, top_p=0.9, seed_rng=7)
    a = dream(model, lens, SEED, **kw)
    b = dream(model, lens, SEED, **kw)
    assert a.surface_text == b.surface_text


def test_concepts_ranked_and_bounded(model, lens):
    result = dream(
        model, lens, SEED, n_steps=20, layers=[1, 2], temperature=0.0,
        consolidate_top_n=5,
    )
    saliences = [c.salience for c in result.concepts]
    assert saliences == sorted(saliences, reverse=True)  # most salient first
    assert len(result.concepts) <= 5
    for c in result.concepts:
        assert isinstance(c, Concept)
        assert 0 <= c.n_latent <= len(result.steps)
        assert 0 <= c.n_surface <= len(result.steps)
        assert c.layers and set(c.layers) <= {1, 2}


def test_spoken_discount_rewards_unspoken(model):
    # Tokens 5 and 9 carry identical rank-weighted latent mass (each is rank 0
    # in one layer and rank 1 in the other), but 5 is always the spoken token.
    # discount < 1 must break the tie in the unspoken token's favour; discount
    # == 1 must restore the tie.
    steps = [
        DreamStep(surface_id=5, surface_token="e", latent={1: [5, 9], 2: [9, 5]})
        for _ in range(6)
    ]
    discounted = {c.token_id: c.salience for c in consolidate(steps, model)}
    assert discounted[9] > discounted[5]  # unspoken beats spoken on equal mass
    flat = {
        c.token_id: c.salience
        for c in consolidate(steps, model, spoken_discount=1.0)
    }
    assert flat[9] == pytest.approx(flat[5])  # discount off -> tie


def test_remember_appends_atomically(tmp_path, model, lens):
    path = tmp_path / "dreams.json"
    r1 = dream(model, lens, SEED, n_steps=8, layers=[1, 2], temperature=0.0)
    remember(r1, path, timestamp=1000.0)
    remember(r1, path, timestamp=2000.0)  # second dream -> appended, not clobbered
    records = json.loads(path.read_text())
    assert isinstance(records, list) and len(records) == 2
    assert records[0]["timestamp"] == 1000.0
    assert records[0]["concepts"] and "salience" in records[0]["concepts"][0]
    assert records[0]["seed"] == SEED


def test_remember_rejects_non_list_file(tmp_path, model, lens):
    path = tmp_path / "bad.json"
    path.write_text('{"not": "a list"}')
    r = dream(model, lens, SEED, n_steps=4, layers=[1, 2], temperature=0.0)
    with pytest.raises(ValueError):
        remember(r, path)


def test_sliding_window_keeps_generating_and_keeps_bos(model, lens):
    # max_seq_len smaller than seed+steps forces the window to slide.
    result = dream(
        model, lens, SEED, n_steps=30, layers=[1, 2], temperature=0.0, max_seq_len=12
    )
    assert len(result.steps) == 30  # never crashes on context overflow
    # The window must keep the BOS attention sink: a slid context is
    # [BOS, <last max_seq_len-1 tokens>], never a bare tail. Assert via the
    # same seam dream() uses, so a regression to plain tail-slicing fails here.
    bos = model.tokenizer.bos_token_id
    seen: list[list[int]] = []

    def spying_forward(input_ids):
        seen.append(input_ids[0].tolist())
        return type(model).forward(model, input_ids)

    model.forward = spying_forward
    try:
        dream(model, lens, SEED, n_steps=8, layers=[1], temperature=0.0, max_seq_len=12)
    finally:
        del model.forward
    overflowed = [ids for ids in seen if len(ids) == 12]
    assert overflowed, "window never slid; lower max_seq_len"
    assert all(ids[0] == bos for ids in overflowed)


def test_bad_n_steps_raises(model, lens):
    with pytest.raises(ValueError):
        dream(model, lens, SEED, n_steps=0, layers=[1, 2])


def test_unknown_layer_raises(model, lens):
    with pytest.raises(ValueError):
        dream(model, lens, SEED, n_steps=4, layers=[3])  # not in source_layers


def test_resolve_layers_defaults_to_subset(lens):
    assert set(_resolve_layers(lens, None)) <= set(lens.source_layers)
    assert _resolve_layers(lens, [1]) == [1]


def test_sample_greedy_and_filtered():
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    assert _sample(logits, temperature=0.0, top_k=0, top_p=1.0, generator=None) == 1
    g = torch.Generator().manual_seed(0)
    # top_k=1 collapses to the argmax even when sampling.
    picks = {
        _sample(logits, temperature=1.0, top_k=1, top_p=1.0, generator=g)
        for _ in range(10)
    }
    assert picks == {1}
