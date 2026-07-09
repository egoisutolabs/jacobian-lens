# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Dreaming: introspective free-generation via the Jacobian lens.

An agent *dreams* by generating freely (offline, no task) while reading its own
residual stream with the lens at each step. The surface token is what the model
"says"; the lens readout at mid layers is what it is *disposed toward but does
not emit* — the road not taken (the ASCII-face "nose" that never appears in the
output). Consolidation accumulates those unspoken latent concepts across the
dream into a ranked list of salient themes, which :func:`remember` distils into
a persistent memory file.

Why this is not just noise: the lens reads intermediate layers, which carry the
semantic attractor the generation is orbiting even when high-temperature
sampling scatters the surface token. Salience deliberately *down-weights*
concepts that were actually spoken (``spoken_discount``), so the consolidated
output is the latent associative neighbourhood of the seed, not a summary of the
surface text.

Caveats. ``J_l`` is a *linear average* Jacobian fit on in-distribution text;
free generation that drifts into the tails degrades the readout. Seed dreams
from real memory, keep ``temperature`` moderate, and keep them short. The lens
only reads the *verbalisable* component (the paper's thesis) — non-verbalisable
computation is invisible here.

    from jlens.dream import dream, remember
    result = dream(model, lens, "I keep coming back to the empty house", n_steps=64)
    print(result)                       # DreamResult(..., top concepts=...)
    remember(result, "out/dreams.json") # append to the agent's memory

CLI: ``python -m jlens.dream --model org/model --lens org/lens-repo \\
        --seed "..." --memory out/dreams.json``
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DreamStep:
    """One generated position.

    Attributes:
        surface_id: Token id the model actually emitted here.
        surface_token: Its decoded string.
        latent: ``{layer: [token_id, ...]}`` — the lens top-k readout at each
            introspected layer, highest-ranked first. The dream content.
    """

    surface_id: int
    surface_token: str
    latent: dict[int, list[int]]


@dataclass(frozen=True)
class Concept:
    """A consolidated latent theme surfaced across the dream.

    Attributes:
        token_id: The vocabulary token.
        text: Its decoded string.
        salience: Rank-weighted latent frequency, with spoken occurrences
            down-weighted (see :func:`consolidate`). Higher = more dwelt-on.
        n_latent: Number of steps it appeared in *any* layer's readout.
        n_surface: Number of steps it was the emitted token (0 = never spoken).
        layers: Introspected layers it surfaced at.
    """

    token_id: int
    text: str
    salience: float
    n_latent: int
    n_surface: int
    layers: list[int]


@dataclass(frozen=True)
class DreamResult:
    """A completed dream: the surface text, the per-step trace, the themes.

    Attributes:
        seed: The prompt the dream started from.
        surface_text: The generated tokens decoded together.
        layers: The layers introspected at each step.
        steps: Per-position :class:`DreamStep`.
        concepts: Consolidated latent themes, most salient first.
    """

    seed: str
    surface_text: str
    layers: list[int]
    steps: list[DreamStep]
    concepts: list[Concept]

    def __repr__(self) -> str:
        top = ", ".join(f"{c.text!r}" for c in self.concepts[:5])
        return (
            f"DreamResult(seed={self.seed[:32]!r}..., steps={len(self.steps)}, "
            f"top concepts=[{top}])"
        )


def _resolve_layers(
    lens: JacobianLens, layers: Sequence[int] | None
) -> list[int]:
    """Layers to introspect at: a validated subset of ``lens.source_layers``.

    Defaults to the middle three fitted layers, where the lens is typically
    most informative (early layers are near the input, late near the output).
    """
    source = lens.source_layers
    if layers is None:
        if len(source) <= 3:
            return list(source)
        mid = len(source) // 2
        return list(source[mid - 1 : mid + 2])
    chosen = sorted(set(layers))
    unknown = [l for l in chosen if l not in set(source)]
    if unknown:
        raise ValueError(
            f"layers {unknown} not in lens.source_layers {source}"
        )
    return chosen


def _sample(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator | None,
) -> int:
    """Sample one token id from ``[vocab]`` logits.

    ``temperature <= 0`` is greedy (deterministic). ``top_k``/``top_p`` filter
    before sampling; both always keep at least the top token.
    """
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        ordered, order = torch.sort(probs, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()  # shift: always keep the top token
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, order, ordered)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, num_samples=1, generator=generator))


@torch.no_grad()
def _readout(
    model: LensModel, input_ids: torch.Tensor, lens: JacobianLens, layers: Sequence[int]
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """One forward pass; return ``(surface_logits, {layer: latent_logits})`` at
    the last position. Mirrors :meth:`JacobianLens.apply` at token level so the
    dream never round-trips ids through text."""
    final = model.n_layers - 1
    record_at = sorted({*layers, final})
    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward(input_ids)
        acts = {i: recorder.activations[i].detach() for i in record_at}
    surface = model.unembed(acts[final][0, -1:].float())[0].cpu()
    latent = {
        layer: model.unembed(lens.transport(acts[layer][0, -1:].float(), layer))[0].cpu()
        for layer in layers
    }
    return surface, latent


def consolidate(
    steps: Sequence[DreamStep],
    model: LensModel,
    *,
    top_n: int = 12,
    spoken_discount: float = 0.4,
) -> list[Concept]:
    """Rank the dream's recurring latent concepts.

    Each latent readout contributes rank-weighted mass to its tokens (top of a
    layer's list counts more). A token's mass at a step where it was *also* the
    emitted token is scaled by ``spoken_discount`` (< 1), so the ranking favours
    the unspoken — the associations the surface suppressed. Set
    ``spoken_discount=1.0`` to score latent frequency without that preference.
    """
    decode = model.tokenizer.decode
    score: dict[int, float] = {}
    n_latent: dict[int, int] = {}
    n_surface: dict[int, int] = {}
    seen_layers: dict[int, set[int]] = {}
    for step in steps:
        n_surface[step.surface_id] = n_surface.get(step.surface_id, 0) + 1
        appeared: set[int] = set()
        for layer, ids in step.latent.items():
            width = len(ids)
            for rank, tid in enumerate(ids):
                weight = width - rank  # rank 0 (top) worth the most
                if tid == step.surface_id:
                    weight *= spoken_discount
                score[tid] = score.get(tid, 0.0) + weight
                seen_layers.setdefault(tid, set()).add(layer)
                appeared.add(tid)
        for tid in appeared:
            n_latent[tid] = n_latent.get(tid, 0) + 1
    ranked = sorted(score, key=lambda t: score[t], reverse=True)[:top_n]
    return [
        Concept(
            token_id=tid,
            text=decode([tid]),
            salience=round(score[tid], 3),
            n_latent=n_latent.get(tid, 0),
            n_surface=n_surface.get(tid, 0),
            layers=sorted(seen_layers[tid]),
        )
        for tid in ranked
    ]


def dream(
    model: LensModel,
    lens: JacobianLens,
    seed: str,
    *,
    n_steps: int = 48,
    layers: Sequence[int] | None = None,
    temperature: float = 0.9,
    top_k: int = 0,
    top_p: float = 1.0,
    latent_topk: int = 8,
    seed_rng: int | None = None,
    max_seq_len: int = 512,
    spoken_discount: float = 0.4,
    consolidate_top_n: int = 12,
) -> DreamResult:
    """Free-generate ``n_steps`` tokens from ``seed`` while lensing each step.

    Args:
        model: A local :class:`~jlens.protocol.LensModel` (must be hookable —
            the residual stream is read directly).
        lens: A fitted :class:`~jlens.lens.JacobianLens` for ``model``.
        seed: Starting text. Seed from real memory to stay in-distribution.
        n_steps: Tokens to generate.
        layers: Layers to introspect at (see :func:`_resolve_layers`).
        temperature: Sampling temperature; ``<= 0`` is greedy/deterministic.
        top_k: Keep only the ``top_k`` logits before sampling (0 = off).
        top_p: Nucleus threshold (1.0 = off).
        latent_topk: Tokens to record from each layer's lens readout per step.
        seed_rng: If set, seeds a CPU generator for reproducible sampling.
        max_seq_len: Context cap; once exceeded the oldest tokens are dropped
            (a sliding window keeps recent context for coherence).
        spoken_discount: See :func:`consolidate`.
        consolidate_top_n: Number of themes to keep.

    Returns:
        A :class:`DreamResult`.

    Raises:
        ValueError: If ``n_steps <= 0`` or a layer is not in ``source_layers``.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    layers = _resolve_layers(lens, layers)
    decode = model.tokenizer.decode
    generator = (
        None if seed_rng is None else torch.Generator().manual_seed(seed_rng)
    )

    input_ids = model.encode(seed, max_length=max_seq_len)
    device = input_ids.device
    steps: list[DreamStep] = []
    for _ in range(n_steps):
        # ponytail: re-forwards the whole context each step -> O(n^2); no KV
        # cache. Fine for short dreams; add caching if n_steps gets large.
        surface_logits, latent_logits = _readout(model, input_ids, lens, layers)
        token_id = _sample(
            surface_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        steps.append(
            DreamStep(
                surface_id=token_id,
                surface_token=decode([token_id]),
                latent={
                    layer: latent_logits[layer].topk(latent_topk).indices.tolist()
                    for layer in layers
                },
            )
        )
        nxt = torch.tensor([[token_id]], device=device)
        input_ids = torch.cat([input_ids, nxt], dim=1)
        if input_ids.shape[1] > max_seq_len:
            # Slide the window but keep position 0: without the BOS attention
            # sink both generation and the lens readout degrade (see force_bos
            # in jlens.hf and skip_first in jlens.fitting).
            input_ids = torch.cat(
                [input_ids[:, :1], input_ids[:, -(max_seq_len - 1) :]], dim=1
            )

    surface_text = decode([s.surface_id for s in steps])
    concepts = consolidate(
        steps, model, top_n=consolidate_top_n, spoken_discount=spoken_discount
    )
    logger.info(
        "dream: %d steps from seed=%r -> top concepts %s",
        n_steps,
        seed[:40],
        [c.text for c in concepts[:5]],
    )
    return DreamResult(
        seed=seed,
        surface_text=surface_text,
        layers=list(layers),
        steps=steps,
        concepts=concepts,
    )


def remember(
    result: DreamResult,
    path: str | Path,
    *,
    timestamp: float | None = None,
    extra: dict | None = None,
) -> dict:
    """Append the dream's consolidated concepts to a JSON memory file.

    The file is a JSON list of records; the write is atomic (temp file +
    ``os.replace``) so a crash never corrupts existing memory. Pass
    ``timestamp`` (e.g. ``time.time()``) to stamp the record; omit it for
    reproducible output. Note JSON stringifies dict keys, so any integer
    layer keys in the record come back as strings on re-load.

    Returns:
        The record that was appended.

    Raises:
        ValueError: If ``path`` exists but does not hold a JSON list.
    """
    record: dict = {
        "seed": result.seed,
        "surface_text": result.surface_text,
        "layers": result.layers,
        "n_steps": len(result.steps),
        "concepts": [asdict(c) for c in result.concepts],
    }
    if timestamp is not None:
        record["timestamp"] = timestamp
    if extra:
        record["extra"] = extra

    path = Path(path)
    records: list = []
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path} is not a JSON list of dream records")
    records.append(record)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp_path, path)
    return record


def _main(argv: Sequence[str] | None = None) -> None:
    import argparse
    import time

    parser = argparse.ArgumentParser(
        prog="python -m jlens.dream",
        description="Dream (introspective free-generation) with a Jacobian lens.",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model id or path")
    parser.add_argument("--lens", required=True, help="lens file, dir, or Hub repo id")
    parser.add_argument(
        "--lens-file", default="lens.pt", help="filename inside a lens dir/repo"
    )
    parser.add_argument("--device", default="cuda", help="device to load the model on")
    parser.add_argument("--seed", required=True, help="text to dream from")
    parser.add_argument("--steps", type=int, default=48, help="tokens to generate")
    parser.add_argument(
        "--layers", type=int, nargs="+", help="layers to introspect (default: middle 3)"
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rng", type=int, default=None, help="seed for reproducibility")
    parser.add_argument("--memory", help="append the dream to this JSON memory file")
    args = parser.parse_args(argv)

    import transformers

    import jlens

    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(args.lens, filename=args.lens_file)

    result = dream(
        model,
        lens,
        args.seed,
        n_steps=args.steps,
        layers=args.layers,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed_rng=args.rng,
    )
    print(result)
    print("\nsurface:", repr(result.surface_text))
    print("\nconsolidated dream (salient latent concepts):")
    for c in result.concepts:
        spoken = "" if c.n_surface == 0 else f", spoken x{c.n_surface}"
        print(f"  {c.text!r:>16}  salience={c.salience:<7} latent x{c.n_latent}{spoken}")
    if args.memory:
        remember(result, args.memory, timestamp=time.time())
        print(f"\nremembered -> {args.memory}")


if __name__ == "__main__":
    _main()
