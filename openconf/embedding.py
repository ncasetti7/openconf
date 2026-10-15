"""Conformer embedding backends."""

from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom

from ._nvmolkit import precision_kwargs
from ._runtime import _NVMOLKIT_EMBED_IMPORTED_BEFORE_RDKIT
from .exceptions import OpenConfValueError

if TYPE_CHECKING:
    from collections.abc import Callable


class ConformerEmbedder(Protocol):
    """Embed conformers into RDKit molecules."""

    def embed(self, mol: Chem.Mol, n_confs: int, params: rdDistGeom.EmbedParameters) -> list[int]:
        """Embed conformers and return their identifiers."""
        ...

    def embed_many(
        self,
        mols: list[Chem.Mol],
        n_confs: int,
        params: rdDistGeom.EmbedParameters,
    ) -> list[list[int]]:
        """Embed conformers into multiple molecules."""
        ...


@dataclass(frozen=True, slots=True)
class RDKitEmbedder:
    """Embed conformers with RDKit."""

    def embed(self, mol: Chem.Mol, n_confs: int, params: rdDistGeom.EmbedParameters) -> list[int]:
        """Embed conformers and return their identifiers."""
        return list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))

    def embed_many(
        self,
        mols: list[Chem.Mol],
        n_confs: int,
        params: rdDistGeom.EmbedParameters,
    ) -> list[list[int]]:
        """Embed conformers into multiple molecules."""
        return [self.embed(mol, n_confs, params) for mol in mols]


@dataclass(frozen=True, slots=True)
class NvMolKitEmbedder:
    """Embed conformers with nvMolKit."""

    precision: str = "full"

    def embed(self, mol: Chem.Mol, n_confs: int, params: rdDistGeom.EmbedParameters) -> list[int]:
        """Embed conformers and return their identifiers.

        Raises:
            OpenConfValueError: nvMolKit is unavailable, unsafe to call, or incompatible with molecule
        """
        return self.embed_many([mol], n_confs, params)[0]

    def embed_many(
        self,
        mols: list[Chem.Mol],
        n_confs: int,
        params: rdDistGeom.EmbedParameters,
    ) -> list[list[int]]:
        """Embed conformers into multiple molecules in one nvMolKit call.

        Raises:
            OpenConfValueError: nvMolKit is unavailable, unsafe to call, or incompatible with molecules
        """
        if not mols:
            return []
        if _NVMOLKIT_EMBED_IMPORTED_BEFORE_RDKIT:
            raise OpenConfValueError(
                "nvMolKit 0.6 ETKDG is unsafe when imported before RDKit. Restart Python and import openconf or "
                "rdkit before nvmolkit.embedMolecules."
            )
        if any(len(Chem.GetMolFrags(mol)) > 1 for mol in mols):
            raise OpenConfValueError("nvMolKit embedding does not support separate fragment embedding.")

        try:
            module = import_module("nvmolkit.embedMolecules")
        except ImportError as exc:
            raise OpenConfValueError("backend='nvmolkit' requires a compatible nvMolKit installation.") from exc

        embed_molecules = cast("Callable[..., None]", module.EmbedMolecules)
        params.useRandomCoords = True
        embed_molecules(
            molecules=mols,
            params=params,
            confsPerMolecule=n_confs,
            maxIterations=-1,
            **precision_kwargs(self.precision),
        )
        return [[conformer.GetId() for conformer in mol.GetConformers()] for mol in mols]


def get_embedder(name: str, precision: str = "full") -> ConformerEmbedder:
    """Return conformer embedder for a backend name.

    Args:
        name: backend name
        precision: nvMolKit arithmetic precision

    Returns:
        Conformer embedding backend

    Raises:
        OpenConfValueError: backend name is unsupported
    """
    if name == "rdkit":
        return RDKitEmbedder()
    if name == "nvmolkit":
        return NvMolKitEmbedder(precision=precision)
    raise OpenConfValueError(f"Unsupported compute backend: {name!r}.")
