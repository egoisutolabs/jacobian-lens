# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Lens-quality evaluation (§methods-comparison).

Runs a fitted lens over one of the ``data/evaluations/*.json`` prompt sets and
reports **pass@k**: the mean over items of the fraction of an item's
``intermediates`` whose best (min-over-layers) lens rank at the readout
position is ``<= k``.

Readout is at a single token position, taken over *all* fitted layers (not a
band). The position is the last prompt token for every set except poetry,
which reads at the last newline (end of line 1 of the couplet). An
``intermediate`` word is scored by the best rank over its single-token surface
forms (bare / leading-space / capitalized); a word with no single-token form
can never be a hit. See ``data/evaluations/README.md`` for the per-set
definitions.

    from jlens.eval import run_eval
    result = run_eval(model, lens, "lens-eval-typo")
    print(result)  # EvalResult(lens-eval-typo, n=..., pass@1=..., pass@5=..., pass@10=...)

CLI: ``python -m jlens.eval lens-eval-typo --model org/model --lens org/lens-repo``
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from jlens.lens import JacobianLens
from jlens.protocol import LensModel

#: pass@k thresholds reported by default.
DEFAULT_KS: tuple[int, ...] = (1, 5, 10)

#: Repo-local eval sets, so a bare slug (``"lens-eval-typo"``) resolves without
#: a full path. Only present in a source checkout; installed users pass a path.
_EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "evaluations"


@dataclass(frozen=True)
class EvalResult:
    """Result of :func:`run_eval` for one eval set.

    Attributes:
        name: Eval-set slug.
        n_items: Number of scored items.
        pass_at_k: ``{k: pass@k}`` for each requested ``k``.
        item_scores: Per-item fraction of intermediates that hit at ``min(ks)``
            — the same fraction ``pass@min(ks)`` averages — for drilldown.
        best_ranks: ``[{intermediate: best_rank_or_None}]`` per item; ``None``
            marks an intermediate with no single-token surface form.
    """

    name: str
    n_items: int
    pass_at_k: dict[int, float]
    item_scores: list[float]
    best_ranks: list[dict[str, int | None]]

    def __repr__(self) -> str:
        ks = ", ".join(f"pass@{k}={self.pass_at_k[k]:.3f}" for k in sorted(self.pass_at_k))
        return f"EvalResult({self.name}, n={self.n_items}, {ks})"


def load_eval(source: str | Path | dict) -> tuple[str, list[dict]]:
    """Resolve ``source`` to ``(name, items)``.

    ``source`` may be an already-loaded ``{"items": [...]}`` dict, a path to
    such a JSON file, or a bare slug resolved against the repo's
    ``data/evaluations/`` directory.
    """
    if isinstance(source, dict):
        return source.get("name", "eval"), source["items"]
    path = Path(source)
    if not path.exists():
        candidate = _EVAL_DIR / f"{path.name}.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"no eval set at {path} or {candidate}; pass a path or a slug "
                f"present in {_EVAL_DIR}"
            )
        path = candidate
    data = json.loads(path.read_text(encoding="utf-8"))
    return path.stem, data["items"]


def _single_token_ids(tokenizer, word: str) -> list[int]:
    """Token ids for surface forms of ``word`` that encode to exactly one token.

    Tries the bare word, a leading-space variant, and capitalized versions of
    both, so a hit does not hinge on which form the tokenizer happens to make
    single-token. Empty when no form is single-token.

    ponytail: covers casing/leading-space only. Full symbol<->word synonym
    expansion (e.g. order-ops "*" == "multiplication") is not applied; pass
    pre-expanded intermediates in the eval JSON if you need it.
    """
    seen: set[int] = set()
    out: list[int] = []
    for form in (word, f" {word}", word.capitalize(), f" {word.capitalize()}"):
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in seen:
            seen.add(ids[0])
            out.append(ids[0])
    return out


def _readout_position(name: str, input_ids: torch.Tensor, tokenizer) -> int:
    """Absolute index of the readout position for eval set ``name``.

    Last prompt token for every set except poetry, which reads at the last
    newline token (end of line 1 of the couplet). See the module docstring.
    """
    ids = input_ids[0].tolist()
    if "poetry" in name:
        newlines = [i for i, t in enumerate(ids) if "\n" in tokenizer.decode([t])]
        if not newlines:
            raise ValueError(f"{name}: prompt has no newline token to read out at")
        return newlines[-1]
    return len(ids) - 1


def _min_rank(lens_logits: dict[int, torch.Tensor], target_ids: torch.Tensor) -> int:
    """Best (smallest) rank of any ``target_ids`` token over all layers, at the
    single position in ``lens_logits`` (each value is ``[1, vocab]``). Rank 0 =
    top; ties count as the number of *strictly* greater logits."""
    best = None
    for logits in lens_logits.values():
        row = logits[0]
        thresholds = row[target_ids]
        ranks = (row.unsqueeze(0) > thresholds.unsqueeze(1)).sum(dim=1)
        r = int(ranks.min())
        best = r if best is None else min(best, r)
    assert best is not None  # apply() always returns >= 1 layer
    return best


@torch.no_grad()
def run_eval(
    model: LensModel,
    lens: JacobianLens,
    source: str | Path | dict,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    max_seq_len: int = 512,
) -> EvalResult:
    """Score ``lens`` on an eval set and return an :class:`EvalResult`.

    Args:
        model: The model to read out from.
        lens: A fitted :class:`~jlens.lens.JacobianLens`.
        source: Eval set — a slug, a path to a ``{"items": [...]}`` JSON file,
            or an already-loaded dict. See :func:`load_eval`.
        ks: pass@k thresholds to report.
        max_seq_len: Truncate each prompt to this many tokens.

    Returns:
        The :class:`EvalResult`; see its docstring for the metric.
    """
    name, items = load_eval(source)
    ks = sorted(ks)
    tokenizer = model.tokenizer

    best_ranks: list[dict[str, int | None]] = []
    for item in items:
        input_ids = model.encode(item["prompt"], max_length=max_seq_len)
        pos = _readout_position(name, input_ids, tokenizer)
        lens_logits, _, _ = lens.apply(
            model, item["prompt"], positions=[pos], max_seq_len=max_seq_len
        )
        ranks: dict[str, int | None] = {}
        for word in item["intermediates"]:
            ids = _single_token_ids(tokenizer, word)
            ranks[word] = (
                _min_rank(lens_logits, torch.tensor(ids)) if ids else None
            )
        best_ranks.append(ranks)

    pass_at_k: dict[int, float] = {}
    for k in ks:
        per_item = [
            sum(r is not None and r < k for r in ranks.values()) / len(ranks)
            for ranks in best_ranks
            if ranks
        ]
        pass_at_k[k] = sum(per_item) / len(per_item) if per_item else 0.0

    k0 = ks[0]
    item_scores = [
        sum(r is not None and r < k0 for r in ranks.values()) / len(ranks)
        for ranks in best_ranks
        if ranks
    ]
    return EvalResult(
        name=name,
        n_items=len(item_scores),
        pass_at_k=pass_at_k,
        item_scores=item_scores,
        best_ranks=best_ranks,
    )


def _main(argv: Sequence[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m jlens.eval", description="Score a Jacobian lens on an eval set."
    )
    parser.add_argument("eval", help="eval-set slug or path to a JSON file")
    parser.add_argument("--model", required=True, help="HuggingFace model id or path")
    parser.add_argument("--lens", required=True, help="lens file, dir, or Hub repo id")
    parser.add_argument(
        "--lens-file", default="lens.pt", help="filename inside a lens dir/repo"
    )
    parser.add_argument("--device", default="cuda", help="device to load the model on")
    parser.add_argument(
        "-k", type=int, nargs="+", default=list(DEFAULT_KS), help="pass@k thresholds"
    )
    args = parser.parse_args(argv)

    import transformers

    import jlens

    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)
    lens = JacobianLens.from_pretrained(args.lens, filename=args.lens_file)
    print(run_eval(model, lens, args.eval, ks=args.k))


if __name__ == "__main__":
    _main()
