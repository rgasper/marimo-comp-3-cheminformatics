"""Central, CWD-independent paths for data the notebook reads.

molab (and any clone) may run the notebook from a different working directory
than the repo root, so we anchor all data paths to the repo root inferred from
this file's location (``src/fingerprints/paths.py`` -> repo root is three
parents up). A ``FINGERPRINTS_DATA`` environment variable can override the data
directory if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/fingerprints/paths.py -> parents[2] == repo root (the dir holding data/)
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("FINGERPRINTS_DATA", REPO_ROOT / "data"))
CACHE_DIR = REPO_ROOT / ".cache"

ADMET_CLIFFS = DATA_DIR / "admet_cliffs" / "admet_cliffs.json"
KNN_CLIFFS = DATA_DIR / "knn_cliffs" / "knn_cliffs.json"
IMPORTANCE = DATA_DIR / "importance" / "rf_importances.json"
LEARNED_CLIFFS = DATA_DIR / "learned_cliffs" / "learned_cliffs.json"
ACCUMULATION = DATA_DIR / "accumulation" / "accumulation.json"
BOLTZ_POSES = DATA_DIR / "boltz_poses"
CHEMELEON_DIR = CACHE_DIR / "chemeleon"
