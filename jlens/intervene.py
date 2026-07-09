# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Writing into the residual stream through the lens (interventions).

Where :meth:`~jlens.lens.JacobianLens.apply` *reads* what a residual is
disposed to make the model say, the helpers here *write*: they add or clamp a
token's lens direction in the residual stream and re-run the model, so a
readout correlation ("the lens shows Paris here") can be turned into a causal
test ("remove Paris and the answer changes").

The lens direction for token ``t`` at layer ``l`` is ``J_l^T @ u_t``, where
``u_t`` is the model's unembedding row for ``t`` (the readout logit is
``u_t . (J_l h)``, so ``J_l^T u_t`` is the direction in residual space that
that readout reads from). :func:`direction` returns it, unit-normalized.

Two convenience moves, both applied at a chosen set of layers and positions:

* :func:`steer` — *add* ``strength * mean_residual_norm_l`` along a token's
  unit direction (the construction in the verbal-introspection /
  directed-modulation experiments).
* :func:`swap` — *move* the residual's coordinate along token A's direction
  onto token B's direction (the "clamp a lens coordinate" swap in the
  probe-swap / verbal-report experiments).

:class:`ResidualEditor` is the low-level context manager both build on; use it
directly for custom edits. Interventions require ``model.unembed_weight`` (see
:class:`~jlens.protocol.LensModel`); fitting and :meth:`apply` do not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from jlens.eval import _single_token_ids
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

#: An edit maps a block's residual output ``[batch, seq, d_model]`` to its
#: replacement.
Edit = Callable[[torch.Tensor], torch.Tensor]


class ResidualEditor:
    """Forward-hook context manager that replaces residual-stream outputs.

    Registers a forward hook on each block in ``edits`` that swaps the block's
    output for ``edit(output)``. Downstream blocks then see the edited stream,
    so the whole rest of the forward pass runs on the intervention. Removes the
    hooks on ``__exit__``.

    Args:
        blocks: The residual blocks (e.g. ``model.layers``).
        edits: ``{block_index: edit}``.
    """

    def __init__(self, blocks: Sequence[nn.Module], edits: dict[int, Edit]) -> None:
        self._blocks = blocks
        self._edits = edits
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., object]:
        edit = self._edits[index]

        def hook(module: nn.Module, inputs, output):
            if torch.is_tensor(output):
                return edit(output)
            # Some HF blocks return a tuple (hidden, present_kv, ...).
            return (edit(output[0]), *output[1:])

        return hook

    def __enter__(self) -> ResidualEditor:
        try:
            for index in self._edits:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _resolve_token(model: LensModel, token: int | str) -> int:
    """A token id from an id or a single-token word/surface form."""
    if isinstance(token, int):
        return token
    ids = _single_token_ids(model.tokenizer, token)
    if not ids:
        raise ValueError(f"{token!r} has no single-token form; pass a token id")
    return ids[0]


def direction(
    lens: JacobianLens,
    model: LensModel,
    token: int | str,
    layer: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Residual-space direction at ``layer`` that the lens reads as ``token``.

    Returns ``J_l^T @ u_t`` (unit-normalized by default), an fp32 ``[d_model]``
    tensor on the CPU. ``token`` is a token id or a single-token word.
    """
    token_id = _resolve_token(model, token)
    if layer not in lens.jacobians:
        raise ValueError(f"layer {layer} not in fitted layers {lens.source_layers}")
    u = model.unembed_weight(token_id).detach().float().cpu()
    d = lens.jacobians[layer].float().t() @ u
    return d / d.norm() if normalize else d


def _positions(positions: Sequence[int] | None, seq_len: int) -> list[int]:
    if positions is None:
        return list(range(seq_len))
    return [p % seq_len for p in positions]


@torch.no_grad()
def _forward_logits(
    model: LensModel, input_ids: torch.Tensor, capture: Sequence[int]
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Run the model and return ``(final_logits[seq_len, vocab], {layer: h[seq_len, d]})``
    for the captured layers. Runs under whatever editor context is active."""
    from jlens.hooks import ActivationRecorder

    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[*capture, final_layer]) as rec:
        model.forward(input_ids)
        acts = {i: rec.activations[i][0].detach() for i in {*capture, final_layer}}
    logits = model.unembed(acts[final_layer]).float().cpu()
    return logits, acts


@torch.no_grad()
def steer(
    lens: JacobianLens,
    model: LensModel,
    prompt: str,
    *,
    token: int | str,
    layers: Sequence[int],
    positions: Sequence[int] | None = None,
    strength: float = 1.0,
    max_seq_len: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add a token's steering direction into the residual and re-run the model.

    At each layer in ``layers``, adds ``strength * mean_residual_norm_l`` along
    ``token``'s unit direction to every position in ``positions`` (all
    positions if ``None``). ``mean_residual_norm_l`` is the mean L2 norm of the
    baseline residual at that layer.

    Returns ``(steered_logits, baseline_logits, input_ids)``; the logits are
    the model's own final-layer output ``[seq_len, vocab]`` with and without
    the intervention. ``strength=0`` reproduces the baseline.
    """
    layers = list(layers)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    pos = _positions(positions, seq_len)

    baseline_logits, acts = _forward_logits(model, input_ids, layers)

    edits: dict[int, Edit] = {}
    for layer in layers:
        unit = direction(lens, model, token, layer)
        mean_norm = acts[layer].norm(dim=-1).mean()
        vec = (strength * mean_norm) * unit
        edits[layer] = _add_at(pos, vec)

    with ResidualEditor(model.layers, edits):
        steered_logits, _ = _forward_logits(model, input_ids, [])
    return steered_logits, baseline_logits, input_ids


@torch.no_grad()
def swap(
    lens: JacobianLens,
    model: LensModel,
    prompt: str,
    *,
    from_: int | str,
    to: int | str,
    layers: Sequence[int],
    positions: Sequence[int] | None = None,
    max_seq_len: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move the residual's ``from_`` lens coordinate onto ``to`` and re-run.

    At each layer/position, with unit directions ``a`` (``from_``) and ``b``
    (``to``): ``h' = h - (h.a) a + (h.a) b`` — the readout mass along ``from_``
    is removed and written onto ``to``. Swapping a token onto itself is the
    identity.

    Returns ``(swapped_logits, baseline_logits, input_ids)``, model final-layer
    logits ``[seq_len, vocab]`` with and without the swap.

    ponytail: this is the "carry A's coordinate onto B, zero A" convention. If
    an experiment needs a different clamp (e.g. set B to a fixed value, or
    project out B first), pass a custom :class:`ResidualEditor` edit instead.
    """
    layers = list(layers)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    pos = _positions(positions, seq_len)

    baseline_logits, _ = _forward_logits(model, input_ids, [])

    edits: dict[int, Edit] = {
        layer: _swap_at(
            pos,
            direction(lens, model, from_, layer),
            direction(lens, model, to, layer),
        )
        for layer in layers
    }
    with ResidualEditor(model.layers, edits):
        swapped_logits, _ = _forward_logits(model, input_ids, [])
    return swapped_logits, baseline_logits, input_ids


def _add_at(positions: Sequence[int], vec: torch.Tensor) -> Edit:
    def edit(residual: torch.Tensor) -> torch.Tensor:
        residual = residual.clone()
        residual[:, positions, :] += vec.to(residual.dtype).to(residual.device)
        return residual

    return edit


def _swap_at(positions: Sequence[int], a_hat: torch.Tensor, b_hat: torch.Tensor) -> Edit:
    def edit(residual: torch.Tensor) -> torch.Tensor:
        residual = residual.clone()
        a = a_hat.to(residual.dtype).to(residual.device)
        b = b_hat.to(residual.dtype).to(residual.device)
        sub = residual[:, positions, :]
        coord = (sub * a).sum(dim=-1, keepdim=True)  # h . a
        residual[:, positions, :] = sub - coord * a + coord * b
        return residual

    return edit
