# Exploring Molecular Fingerprints and Activity Cliffs
*A molab Notebook Competition entry — OpenADMET × marimo*

An interactive marimo notebook that tries to help build intuition for how molecular fingerprints work and what their
limitations are by exploring their mechanics and how they interact with activity cliffs in real chemical and biological datasets.

## Run it

```bash
uv venv --python 3.11
uv pip install -e .
uv run marimo edit fingerprints_notebook.py
```

The first cell fetches the 33 MB CheMeleon weights (once) and then the notebook
is instant. Everything the plots need is **precomputed and shipped** under
`data/`. To rebuild it all from scratch (downloads every source dataset and
re-runs every analysis; ~15 minutes, dominated by the accumulation module's
cross-validation and CheMeleon featurisation), run:

```bash
uv run python -m fingerprints.recompute
```

### Opening a single-file copy (e.g. on molab)

Some ways of opening this notebook — notably molab's "Add from GitHub" with a
direct link to `fingerprints_notebook.py`, or the WebAssembly playground —
only fetch that one file, not the `src/` and `data/` directories it depends
on. The very first cell detects this (by checking whether `fingerprints` is
importable) and, if so, downloads a zip of this repo from GitHub and adds its
`src/` to `sys.path` before anything else runs, so the notebook behaves the
same either way. If you'd rather have molab track your repo directly, use its
GitHub-sync flow with a `blob/<branch>/fingerprints_notebook.py` URL — that
pulls the whole repository tree alongside the notebook.

## What ships precomputed (and how it was made)

| Artifact | File | Built by |
|---|---|---|
| ADMET cliff census (TDC + OpenADMET) | `data/admet_cliffs/` | `fingerprints.analyses.admet` |
| kNN cliff analysis | `data/knn_cliffs/` | `fingerprints.analyses.knn` |
| Feature-importance models | `data/importance/` | `fingerprints.analyses.importance` |
| Binary/count/CheMeleon cross-validation + AqSolDB cliff recovery | `data/accumulation/` | `fingerprints.accumulation` |
| 3D Boltz poses + PLIP interactions | `data/boltz_poses/` | `fingerprints.rebuild_poses` (GPU + `BOLTZ_API_KEY`) |

The first four are rebuilt together by `python -m fingerprints.recompute`
(each is independently runnable too, e.g. `python -m fingerprints.accumulation`).
The Boltz poses need a GPU folding job and a Boltz API key, so they ship as
data (`pip install '.[poses]'` to regenerate).

## Data sources

- **MoleculeACE** (van Tilborg et al. 2022) — curated ChEMBL bioactivities with
  activity-cliff labels; downloaded from the project's GitHub.
- **Therapeutics Data Commons** — AqSolDB aqueous solubility, AstraZeneca
  lipophilicity, Caco-2 permeability.
- **OpenADMET ExpansionRx Challenge** (CC-BY-4.0) — LogD and kinetic solubility
  (KSOL) from the competition host's own released dataset, on Hugging Face.
- **CheMeleon** (Burns et al. 2025) — pretrained message-passing neural
  fingerprint; weights from Zenodo.
- **Boltz-2** predicted complexes + **PLIP** interaction detection (poses framed
  as hypotheses, never experimental structures).

## Chemistry hygiene

All structure handling is RDKit. SMILES are validated; invalid input is handled
gracefully. ADMET data is salt-stripped to the largest organic fragment and
de-duplicated by canonical parent SMILES. Every train/test split is a
**Bemis–Murcko scaffold split** for consistency & simplicity.

## AI disclosure

Built with an AI coding assistant: it helped scaffold cells, the custom `ComplexViewer` anywidget, and
the analyses, and drafted some of the prose. Every chemical claim, data source, and result
was reviewed by a human.
