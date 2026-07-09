# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.dream import Concept, DreamResult, DreamStep, consolidate, dream, remember
from jlens.eval import EvalResult, run_eval
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.intervene import ResidualEditor, direction, steer, swap
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "Concept",
    "DreamResult",
    "DreamStep",
    "EvalResult",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "ResidualEditor",
    "configure_logging",
    "consolidate",
    "direction",
    "dream",
    "fit",
    "from_hf",
    "jacobian_for_prompt",
    "remember",
    "run_eval",
    "steer",
    "swap",
]
