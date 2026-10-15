"""Tests for conformer-generation backends."""

from types import SimpleNamespace
from typing import cast

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from openconf import ConformerConfig, OpenConfValueError, generate_conformers
from openconf.embedding import NvMolKitEmbedder
from openconf.relax import minimize_confs_mmff


def test_nvmolkit_embedding_enables_random_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """NvMolKit embedding enables required random-coordinate generation."""
    observed: dict[str, object] = {}

    def embed_molecules(**kwargs: object) -> None:
        molecules = cast("list[Chem.Mol]", kwargs["molecules"])
        params = cast("rdDistGeom.EmbedParameters", kwargs["params"])
        n_confs = cast("int", kwargs["confsPerMolecule"])
        observed["in_place"] = molecules[0] is mol
        observed["num_threads"] = params.numThreads
        observed["prune_rms_thresh"] = params.pruneRmsThresh
        observed["use_random_coords"] = params.useRandomCoords
        observed["max_iterations"] = kwargs["maxIterations"]
        AllChem.EmbedMultipleConfs(molecules[0], numConfs=n_confs, params=params)

    def import_module(name: str) -> object:
        observed["module"] = name
        return SimpleNamespace(EmbedMolecules=embed_molecules)

    monkeypatch.setattr("openconf.embedding.import_module", import_module)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    params.pruneRmsThresh = 1.5

    conf_ids = NvMolKitEmbedder().embed(mol, 3, params)

    assert observed == {
        "module": "nvmolkit.embedMolecules",
        "in_place": True,
        "num_threads": 0,
        "prune_rms_thresh": 1.5,
        "use_random_coords": True,
        "max_iterations": -1,
    }
    assert conf_ids == [conformer.GetId() for conformer in mol.GetConformers()]
    assert conf_ids


def test_nvmolkit_embedding_rejects_unsafe_import_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject nvMolKit native extension loaded before RDKit."""
    monkeypatch.setattr("openconf.embedding._NVMOLKIT_EMBED_IMPORTED_BEFORE_RDKIT", True)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))

    with pytest.raises(OpenConfValueError, match="imported before RDKit"):
        NvMolKitEmbedder().embed(mol, 1, AllChem.ETKDGv3())


def test_nvmolkit_backend_runs_complete_generation_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """NvMolKit backend routes embedding and minimization through complete generation."""
    calls: list[str] = []

    def embed_molecules(**kwargs: object) -> None:
        molecules = cast("list[Chem.Mol]", kwargs["molecules"])
        params = cast("rdDistGeom.EmbedParameters", kwargs["params"])
        n_confs = cast("int", kwargs["confsPerMolecule"])
        calls.append("embed")
        assert kwargs["maxIterations"] == -1
        assert params.useRandomCoords
        for molecule in molecules:
            assert not molecule.HasProp("_MMFFSanitized")
            assert all(
                atom.GetHybridization() == Chem.HybridizationType.UNSPECIFIED
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() == 1
            )
            AllChem.EmbedMultipleConfs(molecule, numConfs=n_confs, params=params)

    def optimize(**kwargs: object) -> list[list[float]]:
        molecules = cast("list[Chem.Mol]", kwargs["molecules"])
        max_iters = cast("int", kwargs["maxIters"])
        calls.append("optimize")
        assert kwargs["properties"] is not None
        return [
            [energy for _, energy in AllChem.MMFFOptimizeMoleculeConfs(molecule, numThreads=1, maxIters=max_iters)]
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
    config = ConformerConfig(
        backend="nvmolkit",
        max_out=3,
        n_seeds=4,
        n_steps=4,
        num_threads=1,
        pool_max=12,
        random_seed=11,
    )

    ensemble = generate_conformers("CCCC", config=config)

    assert ensemble.records
    assert calls[0] == "embed"
    assert calls.count("optimize") >= 2


def test_nvmolkit_embedding_rejects_disconnected_fragments() -> None:
    """NvMolKit embedding rejects inputs whose fragment treatment would differ."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO.Cl"))

    with pytest.raises(OpenConfValueError):
        NvMolKitEmbedder().embed(mol, 1, AllChem.ETKDGv3())


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
    """NvMolKit minimization reuses configured properties and native energies."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, params=AllChem.ETKDGv3()))
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    assert props is not None
    props.SetMMFFDielectricConstant(10)
    expected = [12.5, 15.25]
    observed: dict[str, object] = {}

    def optimize(**kwargs: object) -> list[list[float]]:
        observed.update(kwargs)
        return [expected]

    def import_module(name: str) -> object:
        observed["module"] = name
        return SimpleNamespace(MMFFOptimizeMoleculesConfs=optimize)

    monkeypatch.setattr("openconf.relax.import_module", import_module)

    energies = minimize_confs_mmff(mol, props, conf_ids[::-1], max_iters=7, backend="nvmolkit")

    assert observed["module"] == "nvmolkit.mmffOptimization"
    assert observed["molecules"] == [mol]
    assert observed["maxIters"] == 7
    properties = cast("list[object]", observed["properties"])
    assert properties[0] is props
    assert energies == pytest.approx(expected[::-1])


def test_nvmolkit_minimization_rejects_misaligned_native_energies(monkeypatch: pytest.MonkeyPatch) -> None:
    """NvMolKit minimization rejects incomplete native energy results."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, params=AllChem.ETKDGv3()))
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    assert props is not None

    module = SimpleNamespace(MMFFOptimizeMoleculesConfs=lambda **kwargs: [[1.0]])
    monkeypatch.setattr("openconf.relax.import_module", lambda name: module)

    with pytest.raises(OpenConfValueError):
        minimize_confs_mmff(mol, props, conf_ids, max_iters=7, backend="nvmolkit")
