# Molecular Fingerprints, Made Tangible

*A molab Notebook Competition entry — OpenADMET × marimo (co-hosted with Pat Walters).*

An interactive marimo notebook that asks one question and chases it all the way
down: **do activity cliffs come from the biology, or from the fingerprint
encoding itself?** Along the way it takes apart MACCS, Morgan, the classical
RDKit fingerprints, and a pretrained **neural** fingerprint (CheMeleon), then
runs the same activity-cliff census on protein-binding data, on public ADMET
benchmarks, and on **OpenADMET's own ExpansionRx challenge data**.

## Run it

```bash
uv venv --python 3.11
uv pip install -e .
uv run marimo edit fingerprints_notebook.py
```

The first cell fetches the 33 MB CheMeleon weights (once) and then the notebook
is instant. Everything the plots need is **precomputed and shipped** under
`data/`. Tick *“Recompute all analyses from scratch”* in the first cell to
rebuild it all live (downloads every source dataset and re-runs the analyses;
~10 minutes, dominated by CheMeleon featurisation).

## What ships precomputed (and how it was made)

| Artifact | File | Built by |
|---|---|---|
| ADMET cliff census (TDC + OpenADMET) | `data/admet_cliffs/` | `fingerprints.analyses.admet` |
| kNN cliff analysis | `data/knn_cliffs/` | `fingerprints.analyses.knn` |
| Feature-importance models | `data/importance/` | `fingerprints.analyses.importance` |
| 3D Boltz poses + PLIP interactions | `data/boltz_poses/` | `fingerprints.rebuild_poses` (GPU + `BOLTZ_API_KEY`) |

The three analyses are re-runnable from the notebook. The Boltz poses need a GPU
folding job and a Boltz API key, so they ship as data (`pip install '.[poses]'`
to regenerate).

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
**Bemis–Murcko scaffold split** so near-duplicate structures never straddle the
split (no leakage).

## AI disclosure

Built in a pairing session with an AI coding assistant, per the competition
guidelines: it helped scaffold cells, the custom `ComplexViewer` anywidget, and
the analyses, and drafted prose. Every chemical claim, data source, and result
was reviewed by a human.
