"""DES-LOC Section 3: Convergence Guarantees verification module.

Claude-2 (M026-M050): Theorem 1 + Step-size + Probabilistic Sync Equivalence.
"""
from .convergence_verifier import (
    PsiParams,
    PsiCalculator,
    StepSizeParams,
    StepSizeValidator,
    ConvergenceParams,
    ConvergenceRateChecker,
    ProbSyncEquivalence,
    TheoremImplicationTests,
    Section3Verifier,
)

__all__ = [
    "PsiParams",
    "PsiCalculator",
    "StepSizeParams",
    "StepSizeValidator",
    "ConvergenceParams",
    "ConvergenceRateChecker",
    "ProbSyncEquivalence",
    "TheoremImplicationTests",
    "Section3Verifier",
]
