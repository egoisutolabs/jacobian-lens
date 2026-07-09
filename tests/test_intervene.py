# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Interventions on the tiny CPU model: direction math, edit invariants."""

from __future__ import annotations

import pytest
import torch

from jlens.fitting import fit
from jlens.intervene import _swap_at, direction, steer, swap

from .tiny import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog"


@pytest.fixture(scope="module")
def model() -> TinyDecoder:
    return TinyDecoder(n_layers=4, d_model=8)


@pytest.fixture(scope="module")
def lens(model):
    return fit(model, ["abcdefghij " * 5], source_layers=[0, 1, 2], dim_batch=4)


def test_direction_is_unit_and_shaped(model, lens):
    d = direction(lens, model, token=5, layer=1)
    assert d.shape == (model.d_model,)
    assert d.norm().item() == pytest.approx(1.0, abs=1e-5)


def test_direction_matches_readout_matrix(model, lens):
    # J_l^T u_t must equal row t of the readout matrix U @ J_l (unnormalized).
    token = 5
    d = direction(lens, model, token=token, layer=1, normalize=False)
    u = model.lm_head.weight[token].float()
    expected = lens.jacobians[1].float().t() @ u
    torch.testing.assert_close(d, expected)


def test_steer_zero_strength_is_baseline(model, lens):
    steered, baseline, _ = steer(
        lens, model, PROMPT, token=5, layers=[1, 2], strength=0.0
    )
    torch.testing.assert_close(steered, baseline)


def test_steer_changes_output(model, lens):
    steered, baseline, _ = steer(
        lens, model, PROMPT, token=5, layers=[1, 2], strength=5.0
    )
    assert not torch.allclose(steered, baseline)


def test_swap_onto_self_is_identity(model, lens):
    swapped, baseline, _ = swap(lens, model, PROMPT, from_=5, to=5, layers=[1, 2])
    torch.testing.assert_close(swapped, baseline)


def test_swap_changes_output(model, lens):
    swapped, baseline, _ = swap(lens, model, PROMPT, from_=5, to=9, layers=[1, 2])
    assert not torch.allclose(swapped, baseline)


def test_swap_edit_moves_coordinate_a_to_b():
    # With orthonormal a, b: the edit must zero the a-coordinate and add the
    # original a-coordinate onto b.
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    h = torch.tensor([[[3.0, 7.0, 2.0]]])  # a-coord=3, b-coord=7
    out = _swap_at([0], a, b)(h)[0, 0]
    assert (out @ a).item() == pytest.approx(0.0, abs=1e-6)  # a zeroed
    assert (out @ b).item() == pytest.approx(7.0 + 3.0)  # b gains a's coord
    assert out[2].item() == pytest.approx(2.0)  # orthogonal component untouched
