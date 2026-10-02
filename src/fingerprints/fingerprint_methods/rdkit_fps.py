"""Classical RDKit fingerprints under a uniform interface.

Includes Morgan, RDKit topological, Atom-pair, Topological torsion, MACCS, Avalon.
All except MACCS use the modern rdFingerprintGenerator API.
"""

from __future__ import annotations

import numpy as np
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import MACCSkeys, Mol, rdFingerprintGenerator

from fingerprints.fingerprint_methods.base import FingerprintResult

# fingerprint length used by all generator-based methods unless noted
DEFAULT_FP_BITS = 2048


def _bits_to_array(bv) -> np.ndarray:
    """Convert an ExplicitBitVect to a uint8 numpy array of 0/1."""
    arr = np.zeros(bv.GetNumBits(), dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(bv, arr)
    return arr


def morgan_fingerprint(
    mols: list[Mol], radius: int = 2, n_bits: int = DEFAULT_FP_BITS
) -> FingerprintResult:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    arr = np.stack([_bits_to_array(gen.GetFingerprint(m)) for m in mols], axis=0)
    return FingerprintResult(
        name=f"Morgan(r={radius},{n_bits}b)",
        kind="binary",
        array=arr,
        n_features=n_bits,
    )


def rdkit_topological_fingerprint(
    mols: list[Mol], n_bits: int = DEFAULT_FP_BITS
) -> FingerprintResult:
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
    arr = np.stack([_bits_to_array(gen.GetFingerprint(m)) for m in mols], axis=0)
    return FingerprintResult(
        name=f"RDKit-topological({n_bits}b)",
        kind="binary",
        array=arr,
        n_features=n_bits,
    )


def atom_pair_fingerprint(
    mols: list[Mol], n_bits: int = DEFAULT_FP_BITS
) -> FingerprintResult:
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    arr = np.stack([_bits_to_array(gen.GetFingerprint(m)) for m in mols], axis=0)
    return FingerprintResult(
        name=f"AtomPair({n_bits}b)",
        kind="binary",
        array=arr,
        n_features=n_bits,
    )


def topological_torsion_fingerprint(
    mols: list[Mol], n_bits: int = DEFAULT_FP_BITS
) -> FingerprintResult:
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
    arr = np.stack([_bits_to_array(gen.GetFingerprint(m)) for m in mols], axis=0)
    return FingerprintResult(
        name=f"TopTorsion({n_bits}b)",
        kind="binary",
        array=arr,
        n_features=n_bits,
    )


def maccs_fingerprint(mols: list[Mol]) -> FingerprintResult:
    arr = np.stack([_bits_to_array(MACCSkeys.GenMACCSKeys(m)) for m in mols], axis=0)
    return FingerprintResult(
        name="MACCS(167b)", kind="binary", array=arr, n_features=arr.shape[1]
    )


def avalon_fingerprint(mols: list[Mol], n_bits: int = 512) -> FingerprintResult:
    arr = np.stack(
        [_bits_to_array(pyAvalonTools.GetAvalonFP(m, nBits=n_bits)) for m in mols],
        axis=0,
    )
    return FingerprintResult(
        name=f"Avalon({n_bits}b)", kind="binary", array=arr, n_features=n_bits
    )


# convenient registry of "all classical" methods
def all_classical(mols: list[Mol]) -> dict[str, FingerprintResult]:
    """Return all classical fingerprints for a list of molecules.

    Returns dict keyed by short stable id (used in plot labels and as dict keys
    in downstream comparison code).
    """
    return {
        "morgan": morgan_fingerprint(mols),
        "rdkit_topo": rdkit_topological_fingerprint(mols),
        "atom_pair": atom_pair_fingerprint(mols),
        "top_torsion": topological_torsion_fingerprint(mols),
        "maccs": maccs_fingerprint(mols),
        "avalon": avalon_fingerprint(mols),
    }
