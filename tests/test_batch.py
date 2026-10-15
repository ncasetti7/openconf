"""Tests for cross-molecule conformer generation."""

from types import SimpleNamespace
from typing import cast

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from openconf import ConformerConfig, OpenConfValueError, generate_conformers_batch


def _install_fake_nvmolkit(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], list[int]]:
    """Install RDKit-backed native API doubles and return call sizes."""
    embedding_call_sizes: list[int] = []
    minimization_call_sizes: list[int] = []

    def embed_molecules(**kwargs: object) -> None:
        molecules = cast("list[Chem.Mol]", kwargs["molecules"])
        params = cast("rdDistGeom.EmbedParameters", kwargs["params"])
        n_confs = cast("int", kwargs["confsPerMolecule"])
        embedding_call_sizes.append(len(molecules))
        for molecule in molecules:
            AllChem.EmbedMultipleConfs(molecule, numConfs=n_confs, params=params)

    def optimize(**kwargs: object) -> list[list[float]]:
        molecules = cast("list[Chem.Mol]", kwargs["molecules"])
        max_iters = cast("int", kwargs["maxIters"])
        minimization_call_sizes.append(len(molecules))
        return [
            [energy for _, energy in AllChem.MMFFOptimizeMoleculeConfs(molecule, maxIters=max_iters)]
            for molecule in molecules
        ]

    def import_module(name: str) -> object:
        if name == "nvmolkit.embedMolecules":
            return SimpleNamespace(EmbedMolecules=embed_molecules)
        if name == "nvmolkit.mmffOptimization":
            return SimpleNamespace(MMFFOptimizeMoleculesConfs=optimize)
        raise ImportError(name)

    monkeypatch.setattr("openconf.embedding.import_module", import_module)
    monkeypatch.setattr("openconf.relax.import_module", import_module)
    return embedding_call_sizes, minimization_call_sizes


def test_batch_uses_single_native_seed_pipeline_and_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compatible mixed inputs share native calls without reordering outputs."""
    embedding_calls, minimization_calls = _install_fake_nvmolkit(monkeypatch)
    inputs: list[Chem.Mol | str] = ["CCO", Chem.MolFromSmiles("CCCC"), "CCN"]
    config = ConformerConfig(
        backend="nvmolkit",
        n_seeds=3,
        max_out=2,
        do_final_refine=False,
        random_seed=17,
        collect_stats=True,
    )

    ensembles = generate_conformers_batch(inputs, config=config)

    smiles = [Chem.MolToSmiles(Chem.RemoveHs(ensemble.mol)) for ensemble in ensembles]
    assert smiles == ["CCO", "CCCC", "CCN"]
    assert all(ensemble.records for ensemble in ensembles)
    assert embedding_calls == [3]
    assert minimization_calls == [3]
    assert [ensemble.generation_stats["seed_embedding_batch_size"] for ensemble in ensembles] == [3, 3, 3]
    assert [ensemble.generation_stats["seed_minimization_batch_size"] for ensemble in ensembles] == [3, 3, 3]


def test_batch_separates_incompatible_embedding_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ring-specific ETKDG settings form separate native embedding groups."""
    embedding_calls, minimization_calls = _install_fake_nvmolkit(monkeypatch)
    config = ConformerConfig(
        backend="nvmolkit",
        n_seeds=2,
        max_out=1,
        n_steps=1,
        do_final_refine=False,
        random_seed=5,
    )

    ensembles = generate_conformers_batch(["CCO", "C1CC1"], config=config)

    assert all(ensemble.records for ensemble in ensembles)
    assert embedding_calls == [1, 1]
    assert minimization_calls[0] == 2


def test_batch_combines_proposal_minimization_across_molecules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid proposal workspaces share one native minimization call."""
    _, minimization_calls = _install_fake_nvmolkit(monkeypatch)
    inputs = ["CCOC1CCCCC1", "CCNC1CCCCC1", "CCCOC1CCCCC1"]
    config = ConformerConfig(
        backend="nvmolkit",
        n_seeds=2,
        max_out=1,
        n_steps=1,
        minimize_batch_size=1,
        move_probs={"single_rotor": 1.0},
        skip_clash_check=True,
        do_final_refine=False,
        random_seed=7,
        collect_stats=True,
    )

    ensembles = generate_conformers_batch(inputs, config=config)

    smiles = [Chem.MolToSmiles(Chem.RemoveHs(ensemble.mol)) for ensemble in ensembles]
    assert smiles == inputs
    assert minimization_calls == [3, 3]
    assert [ensemble.generation_stats["proposal_minimization_batch_size"] for ensemble in ensembles] == [3, 3, 3]


def test_batch_keeps_reference_geometry_inputs_on_single_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing coordinates bypass cross-molecule embedding without changing order."""
    embedding_calls, _ = _install_fake_nvmolkit(monkeypatch)
    reference = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(reference, randomSeed=9)
    config = ConformerConfig(
        backend="nvmolkit",
        n_seeds=2,
        max_out=1,
        do_final_refine=False,
        random_seed=9,
    )

    ensembles = generate_conformers_batch(["CCO", reference], config=config)

    smiles = [Chem.MolToSmiles(Chem.RemoveHs(ensemble.mol)) for ensemble in ensembles]
    assert smiles == ["CCO", "CCCC"]
    assert embedding_calls == [1, 1]


def test_batch_reports_invalid_input_index() -> None:
    """Invalid molecules identify their original batch position."""
    with pytest.raises(OpenConfValueError, match="batch index 1"):
        generate_conformers_batch(["CCO", "not-a-smiles", "CCN"])
