"""Tests for conformer-generation backends."""

from types import SimpleNamespace
from typing import cast

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from openconf import OpenConfValueError
from openconf.embedding import NvMolKitEmbedder
from openconf.relax import minimize_confs_mmff


def test_nvmolkit_embedding_enables_random_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """NvMolKit embedding enables required random-coordinate generation."""
    observed: dict[str, object] = {}

    def embed_molecules(**kwargs: object) -> None:
        molecules = cast("list[Chem.Mol]", kwargs["molecules"])
        params = cast("rdDistGeom.EmbedParameters", kwargs["params"])
        n_confs = cast("int", kwargs["confsPerMolecule"])
        observed["use_random_coords"] = params.useRandomCoords
        observed["max_iterations"] = kwargs["maxIterations"]
        AllChem.EmbedMultipleConfs(molecules[0], numConfs=n_confs, params=params)

    def import_module(name: str) -> object:
        observed["module"] = name
        return SimpleNamespace(EmbedMolecules=embed_molecules)

    monkeypatch.setattr("openconf.embedding.import_module", import_module)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()

    conf_ids = NvMolKitEmbedder().embed(mol, 3, params)

    assert observed == {
        "module": "nvmolkit.embedMolecules",
        "use_random_coords": True,
        "max_iterations": -1,
    }
    assert conf_ids == [conformer.GetId() for conformer in mol.GetConformers()]
    assert conf_ids


def test_nvmolkit_minimization_rejects_partial_conformer_batches() -> None:
    """NvMolKit minimization refuses to mutate unrequested conformers."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, params=params))
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    assert props is not None

    with pytest.raises(OpenConfValueError):
        minimize_confs_mmff(mol, props, conf_ids[:1], max_iters=1, backend="nvmolkit")


def test_nvmolkit_minimization_routes_complete_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """NvMolKit minimization routes complete batches and returns configured energies."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, params=AllChem.ETKDGv3()))
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    assert props is not None
    props.SetMMFFDielectricConstant(10)
    expected = [AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id).CalcEnergy() for conf_id in conf_ids]
    observed: dict[str, object] = {}

    def optimize(**kwargs: object) -> list[list[float]]:
        observed.update(kwargs)
        return [[0 for _ in conf_ids]]

    def import_module(name: str) -> object:
        observed["module"] = name
        return SimpleNamespace(MMFFOptimizeMoleculesConfs=optimize)

    monkeypatch.setattr("openconf.relax.import_module", import_module)

    energies = minimize_confs_mmff(mol, props, conf_ids, max_iters=7, backend="nvmolkit")

    assert observed["module"] == "nvmolkit.mmffOptimization"
    assert observed["molecules"] == [mol]
    assert observed["maxIters"] == 7
    assert observed["properties"] is not props
    assert energies == pytest.approx(expected)
