"""Recompute everything the notebook's precomputed data files contain.

The notebook ships with all analysis outputs precomputed under ``data/`` (see
``fingerprints.paths``) so it loads instantly. This module lets a user
regenerate those files from scratch with **one command**:

    uv run python -m fingerprints.recompute

That downloads the source datasets (MoleculeACE from GitHub, TDC + OpenADMET
ExpansionRx), fetches the CheMeleon weights, and re-runs every analysis in
order. It is intentionally *not* fast — the accumulation module's 5x5
cross-validation and the CheMeleon feature-importance models dominate and push
the whole rebuild to roughly fifteen minutes on a modern CPU.

The 3D Boltz poses are **not** rebuilt here — those need a GPU folding job and
a paid Boltz API key; see ``fingerprints.rebuild_poses`` for that separate,
opt-in path.

Each step is a small callable so both the CLI and the notebook (which wraps
the same sequence in ``mo.status.progress_bar``) can report human-readable
progress.
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


def _run_weights(report: Callable[[str], None]) -> None:
    from fingerprints import chemeleon_fp as chf

    report("downloading CheMeleon weights")
    chf.ensure_weights()


def _run_admet(report: Callable[[str], None]) -> None:
    admet.main(out_dir=paths.ADMET_CLIFFS.parent, on_step=lambda lbl: report(lbl))


def _run_knn(report: Callable[[str], None]) -> None:
    knn.main(out_dir=paths.KNN_CLIFFS.parent, on_step=lambda lbl: report(lbl))


def _run_importance(report: Callable[[str], None]) -> None:
    importance.main(out_dir=paths.IMPORTANCE.parent, on_step=lambda lbl: report(lbl))


def _run_accumulation(report: Callable[[str], None]) -> None:
    from fingerprints import accumulation

    report("binary vs count vs CheMeleon on AqSolDB (5x5 scaffold CV, slow)")
    accumulation.main()


def steps() -> list[Step]:
    """The ordered rebuild steps (weights first — cheap and needed by the RFs)."""
    return [
        Step("CheMeleon weights", _run_weights),
        Step("ADMET cliff census (TDC + OpenADMET)", _run_admet),
        Step("kNN cliff analysis", _run_knn),
        Step("Feature-importance models (CheMeleon — slow)", _run_importance),
        Step("Accumulation + AqSolDB cliff cross-validation (slow)", _run_accumulation),
    ]


def clear_outputs() -> None:
    """Delete the precomputed JSONs so the rebuild is a true from-scratch run."""
    for p in (
        paths.ADMET_CLIFFS,
        paths.KNN_CLIFFS,
        paths.IMPORTANCE,
        paths.ACCUMULATION,
    ):
        if p.exists():
            p.unlink()


def main() -> None:
    """Run every rebuild step from the command line, printing progress."""
    clear_outputs()
    for step in steps():
        print(f"== {step.title} ==")
        step.run(lambda msg: print(f"  {msg}"))
    print("done.")


if __name__ == "__main__":
    main()
