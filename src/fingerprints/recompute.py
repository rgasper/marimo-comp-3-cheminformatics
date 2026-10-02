"""Recompute everything the notebook's precomputed data files contain.

The notebook ships with all analysis outputs precomputed under ``data/`` (see
``fingerprints.paths``) so it loads instantly. This module lets a curious user
regenerate those files from scratch — downloading the source datasets
(MoleculeACE from GitHub, TDC + OpenADMET ExpansionRx), fetching the CheMeleon
weights, and re-running the three analyses.

It is intentionally *not* fast: the CheMeleon featurisation for the
feature-importance models dominates and pushes the whole rebuild to roughly ten
minutes on CPU. The 3D Boltz poses are **not** rebuilt here — those need a GPU
folding job and a Boltz API key; see ``rebuild_poses`` for that separate, opt-in
path.

Each step is a small callable so the notebook can wrap the sequence in
``mo.status.progress_bar`` and report human-readable progress.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from fingerprints import paths
from fingerprints.analyses import admet, importance, knn


@dataclass(frozen=True)
class Step:
    """One rebuild step: a title and the work to run."""

    title: str
    run: Callable[[Callable[[str], None]], None]


def _run_admet(report: Callable[[str], None]) -> None:
    admet.main(out_dir=paths.ADMET_CLIFFS.parent, on_step=lambda lbl: report(lbl))


def _run_knn(report: Callable[[str], None]) -> None:
    knn.main(out_dir=paths.KNN_CLIFFS.parent, on_step=lambda lbl: report(lbl))


def _run_importance(report: Callable[[str], None]) -> None:
    importance.main(out_dir=paths.IMPORTANCE.parent, on_step=lambda lbl: report(lbl))


def _run_learned_cliffs(report: Callable[[str], None]) -> None:
    from fingerprints import learned_cliffs

    report("training {ECFP, CheMeleon} × {linear, kNN, MLP} on binding endpoints")
    learned_cliffs.main()


def _run_weights(report: Callable[[str], None]) -> None:
    from fingerprints import chemeleon_fp as chf

    report("downloading CheMeleon weights")
    chf.ensure_weights()


def steps() -> list[Step]:
    """The ordered rebuild steps (weights first — cheap and needed by the RFs)."""
    return [
        Step("CheMeleon weights", _run_weights),
        Step("ADMET cliff census (TDC + OpenADMET)", _run_admet),
        Step("kNN cliff analysis", _run_knn),
        Step("Feature-importance models (CheMeleon — slow)", _run_importance),
        Step("Learned-fingerprint cliff head-sweep", _run_learned_cliffs),
    ]


def clear_outputs() -> None:
    """Delete the precomputed JSONs so the rebuild is a true from-scratch run."""
    for p in (
        paths.ADMET_CLIFFS,
        paths.KNN_CLIFFS,
        paths.IMPORTANCE,
        paths.LEARNED_CLIFFS,
    ):
        if p.exists():
            p.unlink()
