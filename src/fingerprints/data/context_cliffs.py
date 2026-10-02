"""Curated context-dependent activity cliffs across related target pairs.

A *context-dependent cliff* is a pair of near-identical molecules whose potency
gap is huge on one target but flat on another. It is the sharpest illustration
that an "activity cliff" is not a property of a molecular pair by itself - it is
defined *relative to an endpoint*. That is a core reason cliffs are hard: a
model (or fingerprint) that only sees structure cannot know that the same
one-atom change matters enormously for one target and not at all for another.

These pairs were mined from the MoleculeACE benchmark (van Tilborg et al. 2022)
by an offline all-pairs MCS search over the molecules shared between two related
targets - then **hand-selected** here for chemical legibility. Each pair:

  * differs by a small, real, medicinal-chemistry change (a bioisostere, a
    halogen, one ring atom) - never a tautomer, salt, or stereochemistry-only
    difference;
  * is a genuine cliff (>= ~100x potency gap) on one target and essentially
    flat (< ~10x) on the other;
  * carries pKi values for both targets so the contrast is explicit.

pKi values are from MoleculeACE (originally ChEMBL); higher = more potent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextCliff:
    """One molecule pair, scored on two related targets."""

    smiles_1: str
    smiles_2: str
    change: str  # plain-English description of the structural difference
    target_a: str  # label of the first target
    target_b: str  # label of the second target
    pki_1_a: float  # pKi of molecule 1 on target A
    pki_2_a: float  # pKi of molecule 2 on target A
    pki_1_b: float  # pKi of molecule 1 on target B
    pki_2_b: float  # pKi of molecule 2 on target B

    @property
    def delta_a(self) -> float:
        return abs(self.pki_1_a - self.pki_2_a)

    @property
    def delta_b(self) -> float:
        return abs(self.pki_1_b - self.pki_2_b)

    @property
    def cliff_on(self) -> str:
        return self.target_a if self.delta_a >= self.delta_b else self.target_b

    @property
    def flat_on(self) -> str:
        return self.target_b if self.delta_a >= self.delta_b else self.target_a

    def actual_pki(self, mol_index: int, target: str) -> float:
        """Measured pKi for molecule 1 or 2 (``mol_index`` in {1, 2}) on the
        given target label."""
        if target == self.target_a:
            return self.pki_1_a if mol_index == 1 else self.pki_2_a
        if target == self.target_b:
            return self.pki_1_b if mol_index == 1 else self.pki_2_b
        raise KeyError(target)


@dataclass(frozen=True)
class TargetPair:
    """A pair of related targets and the curated context cliffs between them."""

    key: str
    target_a: str
    target_b: str
    blurb: str
    cliffs: tuple[ContextCliff, ...]


# --- Dopamine D3 vs D4 (both GPCRs; subtype selectivity is a real design goal) ---
D3_D4 = TargetPair(
    key="D3_vs_D4",
    target_a="Dopamine D3",
    target_b="Dopamine D4",
    blurb=(
        "Two closely related dopamine receptor subtypes."
    ),
    cliffs=(
        ContextCliff(
            smiles_1="CCOc1ccccc1N1CCN(Cc2c[nH]c3ccccc23)CC1",
            smiles_2="c1ccc(N2CCN(Cc3c[nH]c4ccccc34)CC2)cc1",
            change="removing the ortho-ethoxy group from the phenyl-piperazine",
            target_a="Dopamine D3",
            target_b="Dopamine D4",
            pki_1_a=6.99, pki_2_a=6.55, pki_1_b=10.52, pki_2_b=8.1,
        ),
        ContextCliff(
            smiles_1="COc1ccc(N2CCN(Cc3c(C)nc4cc(C)nc(C)n34)CC2)cc1",
            smiles_2="Cc1cc2nc(C)c(CN3CCN(c4ccc(F)cc4)CC3)n2c(C)n1",
            change="swapping a methoxyphenyl for a fluorophenyl on the piperazine",
            target_a="Dopamine D3",
            target_b="Dopamine D4",
            pki_1_a=6.59, pki_2_a=10.0, pki_1_b=8.1, pki_2_b=8.53,
        ),
        ContextCliff(
            smiles_1="OC1(c2ccc(Cl)c(Cl)c2)CCN(Cc2c[nH]c3ccccc23)CC1",
            smiles_2="Oc1ccc(C2(O)CCN(Cc3c[nH]c4ccccc34)CC2)cc1",
            change="replacing a 3,4-dichlorophenyl with a 4-hydroxyphenyl",
            target_a="Dopamine D3",
            target_b="Dopamine D4",
            pki_1_a=8.3, pki_2_a=6.12, pki_1_b=7.4, pki_2_b=7.31,
        ),
        ContextCliff(
            smiles_1="CCOc1ccccc1N1CCN(Cc2c[nH]c3ccccc23)CC1",
            smiles_2="CCOc1ccccc1N1CCN(Cc2csc3ccccc23)CC1",
            change="swapping an indole for a benzothiophene (NH -> S)",
            target_a="Dopamine D3",
            target_b="Dopamine D4",
            pki_1_a=6.99, pki_2_a=7.35, pki_1_b=10.52, pki_2_b=8.34,
        ),
    ),
)


# --- mu vs kappa opioid (both GPCRs; classic analgesic selectivity problem) ---
MU_KAPPA = TargetPair(
    key="mu_vs_kappa",
    target_a="mu-opioid",
    target_b="kappa-opioid",
    blurb=(
        "The mu and kappa opioid receptors. On this shared morphinan-like "
        "series, small substituent swaps can be a potency cliff on one receptor "
        "and invisible on the other."
    ),
    cliffs=(
        ContextCliff(
            smiles_1="CO[C@]12CC[C@@]3(C[C@@H]1COCc1ccccc1)[C@H]1Cc4ccc(C#N)c5c4[C@@]3(CCN1CC1CC1)[C@H]2O5",
            smiles_2="CO[C@]12CC[C@@]3(C[C@@H]1COCc1ccccc1)[C@H]1Cc4ccc(C(N)=O)c5c4[C@@]3(CCN1CC1CC1)[C@H]2O5",
            change="a nitrile -> primary amide bioisostere on the aromatic ring",
            target_a="mu-opioid",
            target_b="kappa-opioid",
            pki_1_a=7.08, pki_2_a=9.47, pki_1_b=10.15, pki_2_b=10.0,
        ),
        ContextCliff(
            smiles_1="Cc1cnccc1C(=O)N[C@@H]1CC[C@@]2(O)[C@H]3Cc4ccc(O)c5c4[C@@]2(CCN3CC2CC2)[C@H]1O5",
            smiles_2="O=C(N[C@@H]1CC[C@@]2(O)[C@H]3Cc4ccc(O)c5c4[C@@]2(CCN3CC2CC2)[C@H]1O5)c1ccnc(Br)c1",
            change="moving/adding substituents on the pyridine amide (methyl -> bromo)",
            target_a="mu-opioid",
            target_b="kappa-opioid",
            pki_1_a=9.24, pki_2_a=9.2, pki_1_b=7.01, pki_2_b=9.74,
        ),
        ContextCliff(
            smiles_1="Oc1ccc2c(c1)[C@@]13CCCC[C@H]1[C@@H](C2)N(CC1CC1)CC3",
            smiles_2="Oc1ccc2c(c1)[C@@]13CCNC[C@H]1[C@@H](C2)N(CC1CC1)CC3",
            change="replacing a ring CH2 with an NH in the fused carbocycle",
            target_a="mu-opioid",
            target_b="kappa-opioid",
            pki_1_a=10.21, pki_2_a=7.3, pki_1_b=10.47, pki_2_b=9.68,
        ),
        ContextCliff(
            smiles_1="Oc1ccc(CCN(CCc2ccccc2)CC2CC2)cc1",
            smiles_2="Oc1cccc(CCN(CCc2ccccc2)CC2CCC2)c1",
            change="para -> meta phenol plus cyclopropyl -> cyclobutyl",
            target_a="mu-opioid",
            target_b="kappa-opioid",
            pki_1_a=5.76, pki_2_a=6.27, pki_1_b=6.66, pki_2_b=9.31,
        ),
    ),
)


TARGET_PAIRS: tuple[TargetPair, ...] = (D3_D4, MU_KAPPA)


def by_key() -> dict[str, TargetPair]:
    return {tp.key: tp for tp in TARGET_PAIRS}


def pair_options() -> dict[str, str]:
    """Dropdown options mapping human label -> target-pair key."""
    return {f"{tp.target_a} vs {tp.target_b}": tp.key for tp in TARGET_PAIRS}


def cliff_options(pair_key: str) -> dict[str, int]:
    """Dropdown options mapping cliff label -> index, for one target pair."""
    tp = by_key()[pair_key]
    return {
        f"{c.change}  \u2014  cliff on {c.cliff_on}": i
        for i, c in enumerate(tp.cliffs)
    }


def clamp_cliff_idx(pair_key: str, idx: int) -> int:
    """Keep a cliff index valid when the selected target pair changes."""
    n = len(by_key()[pair_key].cliffs)
    if n == 0:
        return 0
    return max(0, min(idx, n - 1))


def cliff_molecule_options(pair_key: str, idx: int) -> dict[str, str]:
    """Menu of the two molecules in one cliff: display label -> SMILES.

    Lets the "what does a fingerprint encode" section reuse the *same* two
    molecules the reader is already studying in the cliff, instead of an
    unrelated gallery. Labels spell out that these are the exact cliff pair.
    """
    tp = by_key()[pair_key]
    idx = clamp_cliff_idx(pair_key, idx)
    c = tp.cliffs[idx]
    return {
        f"Molecule 1 — before ({c.change})": c.smiles_1,
        f"Molecule 2 — after ({c.change})": c.smiles_2,
    }
