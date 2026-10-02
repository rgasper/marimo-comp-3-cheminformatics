"""Fetch and cache MoleculeACE benchmark datasets.

MoleculeACE (van Tilborg et al. 2022, J. Chem. Inf. Model. 62, 5938-5951)
is a curated activity-cliff benchmark covering 30 ChEMBL targets. Each
dataset is a CSV with columns:

- smiles: molecule SMILES
- exp_mean [nM]: assay value in nM
- y: log10(nM) used by the original paper for training
- cliff_mol: 1 if the molecule is involved in any cliff pair (Tanimoto
  >= 0.9 to some other molecule AND |Delta(pKi)| >= 1.0)
- split: "train" or "test" (the canonical MoleculeACE split)
- y [pEC50/pKi]: shifted to standard pKi/pEC50, == 9 - log10(nM). This is
  what we use as the regression target.

Datasets are pulled from the project's GitHub repo via raw.githubusercontent
and cached locally. We bypass the MoleculeACE pip package because it pins
old versions of TF/PyTorch/transformers that conflict with our env.

Reference: https://github.com/molML/MoleculeACE
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests
from loguru import logger
from tqdm import tqdm
from typeguard import typechecked

from fingerprints.exceptions import DataLoadError

RAW_BASE = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/main/"
    "MoleculeACE/Data/benchmark_data/"
)


@dataclass(frozen=True)
class MolACEDataset:
    """Metadata for one MoleculeACE benchmark dataset.

    Args:
        name: dataset id used in the CSV filename, e.g. "CHEMBL234_Ki".
        target_label: human-readable target name, e.g. "Dopamine D3 receptor".
        target_class: short class label, e.g. "GPCR" / "kinase" / "protease".
        assay_type: "Ki" or "EC50" (the suffix of the dataset name).
    """

    name: str
    target_label: str
    target_class: str
    assay_type: str


# Three targets used in the headline cliffs analysis (kept as named exports
# for backward compatibility with figure_cliffs.py and figure_cliffs_aggregate.py).
D3_DOPAMINE = MolACEDataset(
    name="CHEMBL234_Ki",
    target_label="Dopamine D3 receptor",
    target_class="GPCR",
    assay_type="Ki",
)
THROMBIN = MolACEDataset(
    name="CHEMBL204_Ki",
    target_label="Thrombin (F2)",
    target_class="protease",
    assay_type="Ki",
)
GSK3B = MolACEDataset(
    name="CHEMBL262_Ki",
    target_label="GSK-3 beta",
    target_class="kinase",
    assay_type="Ki",
)


# Full MoleculeACE benchmark target list (30 datasets), reproduced from
# https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/Data/benchmark_data/metadata/datasets.csv
# Used for the cross-target aggregation analysis. target_class follows the
# original "Receptor Class" column (lowercased for consistency with the
# three-target exports above).
ALL_MOLACE_DATASETS: tuple[MolACEDataset, ...] = (
    MolACEDataset(name="CHEMBL1871_Ki", target_label="Androgen Receptor", target_class="NR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL218_EC50", target_label="Cannabinoid receptor 1", target_class="GPCR", assay_type="EC50"),
    MolACEDataset(name="CHEMBL244_Ki", target_label="Coagulation factor X", target_class="protease", assay_type="Ki"),
    MolACEDataset(name="CHEMBL236_Ki", target_label="Delta opioid receptor", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL234_Ki", target_label="Dopamine D3 receptor", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL219_Ki", target_label="Dopamine D4 receptor", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL238_Ki", target_label="Dopamine transporter", target_class="other", assay_type="Ki"),
    MolACEDataset(name="CHEMBL4203_Ki", target_label="Dual specificity protein kinase CLK4", target_class="kinase", assay_type="Ki"),
    MolACEDataset(name="CHEMBL2047_EC50", target_label="Farnesoid X receptor", target_class="NR", assay_type="EC50"),
    MolACEDataset(name="CHEMBL4616_EC50", target_label="Ghrelin receptor", target_class="GPCR", assay_type="EC50"),
    MolACEDataset(name="CHEMBL2034_Ki", target_label="Glucocorticoid receptor", target_class="NR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL262_Ki", target_label="GSK-3 beta", target_class="kinase", assay_type="Ki"),
    MolACEDataset(name="CHEMBL231_Ki", target_label="Histamine H1 receptor", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL264_Ki", target_label="Histamine H3 receptor", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL2835_Ki", target_label="Janus kinase 1", target_class="kinase", assay_type="Ki"),
    MolACEDataset(name="CHEMBL2971_Ki", target_label="Janus kinase 2", target_class="kinase", assay_type="Ki"),
    MolACEDataset(name="CHEMBL237_EC50", target_label="Kappa opioid receptor (EC50)", target_class="GPCR", assay_type="EC50"),
    MolACEDataset(name="CHEMBL237_Ki", target_label="Kappa opioid receptor (Ki)", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL4792_Ki", target_label="Orexin receptor 2", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL239_EC50", target_label="PPAR alpha", target_class="NR", assay_type="EC50"),
    MolACEDataset(name="CHEMBL3979_EC50", target_label="PPAR delta", target_class="NR", assay_type="EC50"),
    MolACEDataset(name="CHEMBL235_EC50", target_label="PPAR gamma", target_class="NR", assay_type="EC50"),
    MolACEDataset(name="CHEMBL4005_Ki", target_label="PI3-kinase p110-alpha", target_class="transferase", assay_type="Ki"),
    MolACEDataset(name="CHEMBL2147_Ki", target_label="PIM1 kinase", target_class="kinase", assay_type="Ki"),
    MolACEDataset(name="CHEMBL214_Ki", target_label="Serotonin 1a receptor", target_class="GPCR", assay_type="Ki"),
    MolACEDataset(name="CHEMBL228_Ki", target_label="Serotonin transporter", target_class="other", assay_type="Ki"),
    MolACEDataset(name="CHEMBL287_Ki", target_label="Sigma opioid receptor", target_class="other", assay_type="Ki"),
    MolACEDataset(name="CHEMBL204_Ki", target_label="Thrombin (F2)", target_class="protease", assay_type="Ki"),
    MolACEDataset(name="CHEMBL1862_Ki", target_label="ABL1 kinase", target_class="kinase", assay_type="Ki"),
    MolACEDataset(name="CHEMBL233_Ki", target_label="mu-opioid receptor", target_class="GPCR", assay_type="Ki"),
)


@typechecked
def download_molace(ds: MolACEDataset, dest: Path) -> Path:
    """Download a MoleculeACE CSV to dest if missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"{ds.name} already present at {dest}")
        return dest
    url = f"{RAW_BASE}{ds.name}.csv"
    logger.info(f"downloading {ds.name} from {url} -> {dest}")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, desc=ds.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except requests.RequestException as e:
        raise DataLoadError(f"failed to download {url}: {e}") from e
    return dest


@typechecked
def load_molace(ds: MolACEDataset, cache_dir: Path) -> pl.DataFrame:
    """Download (if needed) and parse a MoleculeACE benchmark CSV.

    Returns a DataFrame with these standardized columns:

    | column     | dtype | meaning                                            |
    |------------|-------|----------------------------------------------------|
    | smiles     | str   | molecule SMILES                                    |
    | y          | f64   | activity in pKi or pEC50 units (higher = stronger) |
    | cliff_mol  | bool  | True if the molecule sits in any cliff pair        |
    | split      | str   | "train" or "test" per the MoleculeACE canonical    |
    """
    path = cache_dir / f"{ds.name}.csv"
    download_molace(ds, path)
    df = pl.read_csv(path)
    expected = {"smiles", "cliff_mol", "split", "y [pEC50/pKi]"}
    missing = expected - set(df.columns)
    if missing:
        raise DataLoadError(
            f"unexpected MoleculeACE schema for {ds.name}: missing {missing}, "
            f"got columns={df.columns}"
        )
    out = df.select(
        pl.col("smiles"),
        pl.col("y [pEC50/pKi]").cast(pl.Float64, strict=False).alias("y"),
        (pl.col("cliff_mol").cast(pl.Int64, strict=False) == 1).alias("cliff_mol"),
        pl.col("split"),
    ).drop_nulls(["smiles", "y"])
    logger.info(
        f"loaded {ds.name}: {out.height} rows "
        f"({out.filter(pl.col('cliff_mol')).height} cliff mols, "
        f"{out.filter(pl.col('split') == 'test').height} test mols)"
    )
    return out
