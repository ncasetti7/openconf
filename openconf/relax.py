"""Relaxation and minimization backends for conformer generation."""

from dataclasses import dataclass, field
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from rdkit import Chem
from rdkit.Chem import AllChem

from .constraints import ConstraintModel, minimize_with_constraints
from .exceptions import OpenConfValueError

if TYPE_CHECKING:
    from collections.abc import Callable


class Minimizer(Protocol):
    """Protocol for conformer minimizers."""

    max_iters: int

    def minimize(self, mol: Chem.Mol, conf_id: int) -> float:
        """Minimize a conformer in place.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to minimize

        Returns:
            Energy in kcal/mol after minimization
        """
        ...


@dataclass
class RDKitMMFFMinimizer:
    """RDKit MMFF94 force field minimizer.

    Attributes:
        max_iters: maximum iterations for minimization
        force_tol: force convergence tolerance
        energy_tol: energy convergence tolerance
        variant: MMFF variant; `"MMFF94"` or `"MMFF94s"`
        dielectric: dielectric constant for electrostatics. Gas phase is 1.0;
            higher values (4-10) reduce over-strong intramolecular electrostatics
            and are more appropriate for condensed-phase conformer generation.
        metal_atom_indices: metal-center atom indices. When MMFF is
            unavailable these are pinned via UFFAddPositionConstraint so that
            untyped metals (lanthanides, actinides, …) do not drift freely
            during UFF minimization.
        constraint_model: geometry constraints to apply during minimization.
    """

    max_iters: int = 500
    force_tol: float = 1e-4
    energy_tol: float = 1e-6
    variant: str = "MMFF94s"
    dielectric: float = 4.0
    metal_atom_indices: frozenset[int] = field(default_factory=frozenset)
    constraint_model: ConstraintModel = field(default_factory=ConstraintModel.empty)
    _mmff_props: object = field(default=None, init=False, repr=False)
    _prepared_constraints: ConstraintModel = field(default_factory=ConstraintModel.empty, init=False, repr=False)

    def prepare(self, mol: Chem.Mol) -> None:
        """Cache MMFF properties for the molecule.

        Call once per molecule before minimizing.

        Args:
            mol: molecule to prepare
        """
        self._mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=self.variant)
        if self._mmff_props is not None:
            self._mmff_props.SetMMFFDielectricConstant(self.dielectric)
        metal_constraints = ConstraintModel.from_metal_shell(mol, self.metal_atom_indices)
        self._prepared_constraints = metal_constraints.combine(self.constraint_model)

    def minimize(self, mol: Chem.Mol, conf_id: int) -> float:
        """Minimize conformer in place and return energy in kcal/mol.

        Args:
            mol: molecule containing conformer
            conf_id: conformer ID to minimize

        Returns:
            Energy in kcal/mol after minimization
        """
        try:
            result = minimize_with_constraints(
                mol,
                conf_id,
                self._prepared_constraints,
                self.max_iters,
                mmff_props=self._mmff_props,
            )
            return result.energy if result is not None else float("inf")
        except (ValueError, RuntimeError):
            return float("inf")


def minimize_confs_mmff(
    mol: Chem.Mol,
    mmff_props,
    conf_ids: list[int],
    max_iters: int,
    num_threads: int = 0,
    variant: str = "MMFF94s",
    backend: str = "rdkit",
) -> list[float]:
    """Minimize conformers and return energies evaluated with mmff_props.

    Uses selected batch optimizer for geometry minimization, then re-evaluates
    each conformer's energy with pre-prepared properties carrying the custom
    dielectric.

    Note: geometries are at the default-dielectric (ε=1) MMFF minimum; the custom
    dielectric is applied only to the energy evaluation. For the coarse fast-
    minimization passes during MCMM sampling this is an acceptable approximation.

    Args:
        mol: molecule containing conformers
        mmff_props: pre-prepared MMFFMoleculeProperties with dielectric already set
        conf_ids: conformer IDs to minimize
        max_iters: maximum minimization iterations
        num_threads: C++ threads for MMFFOptimizeMoleculeConfs; 0 means all available
        variant: MMFF variant string passed to MMFFOptimizeMoleculeConfs
        backend: compute backend name

    Returns:
        Energies in kcal/mol aligned to `conf_ids`
    """
    return minimize_molecules_mmff(
        [mol],
        [mmff_props],
        [conf_ids],
        max_iters,
        num_threads=num_threads,
        variant=variant,
        backend=backend,
    )[0]


def minimize_molecules_mmff(
    mols: list[Chem.Mol],
    mmff_props: list[object],
    conf_ids: list[list[int]],
    max_iters: int,
    num_threads: int = 0,
    variant: str = "MMFF94s",
    backend: str = "rdkit",
) -> list[list[float]]:
    """Minimize conformers across multiple molecules.

    Args:
        mols: molecules containing conformers
        mmff_props: prepared MMFF properties aligned with `mols`
        conf_ids: conformer IDs aligned with `mols`
        max_iters: maximum minimization iterations
        num_threads: CPU threads used by RDKit
        variant: MMFF variant used for geometry optimization
        backend: compute backend name

    Returns:
        Energies grouped by molecule and aligned with `conf_ids`

    Raises:
        OpenConfValueError: inputs are misaligned or backend is unsupported
    """
    if len(mols) != len(mmff_props) or len(mols) != len(conf_ids):
        raise OpenConfValueError("mols, mmff_props, and conf_ids must have equal lengths.")
    if not mols:
        return []

    if backend == "rdkit":
        for mol, ids in zip(mols, conf_ids, strict=True):
            if ids:
                AllChem.MMFFOptimizeMoleculeConfs(
                    mol,
                    numThreads=int(num_threads or 0),
                    maxIters=int(max_iters),
                    mmffVariant=variant,
                )
    elif backend == "nvmolkit":
        _minimize_molecules_nvmolkit(mols, conf_ids, max_iters, variant)
    else:
        raise OpenConfValueError(f"Unsupported compute backend: {backend!r}.")

    return [
        [
            float(ff.CalcEnergy())
            if (ff := AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)) is not None
            else float("inf")
            for conf_id in ids
        ]
        for mol, props, ids in zip(mols, mmff_props, conf_ids, strict=True)
    ]


def _minimize_molecules_nvmolkit(
    mols: list[Chem.Mol],
    conf_ids: list[list[int]],
    max_iters: int,
    variant: str,
) -> None:
    """Minimize all conformers from multiple molecules with nvMolKit."""
    for mol, ids in zip(mols, conf_ids, strict=True):
        molecule_conf_ids = [conformer.GetId() for conformer in mol.GetConformers()]
        if len(molecule_conf_ids) != len(ids) or set(molecule_conf_ids) != set(ids):
            raise OpenConfValueError(
                "nvMolKit batch minimization requires conf_ids to include every molecule conformer."
            )

    try:
        module = import_module("nvmolkit.mmffOptimization")
    except ImportError as exc:
        raise OpenConfValueError("backend='nvmolkit' requires a compatible nvMolKit installation.") from exc

    optimize = cast("Callable[..., list[list[float]]]", module.MMFFOptimizeMoleculesConfs)
    optimization_props = [AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant) for mol in mols]
    if any(props is None for props in optimization_props):
        raise OpenConfValueError(f"At least one molecule has no {variant} parameters.")
    optimize(
        molecules=mols,
        maxIters=int(max_iters),
        properties=optimization_props,
    )


def get_minimizer(
    name: str = "rdkit_mmff",
    metal_atom_indices: frozenset[int] = frozenset(),
    constraint_model: ConstraintModel | None = None,
    **kwargs,
) -> Minimizer:
    """Get a minimizer by name.

    Args:
        name: minimizer name; only `"rdkit_mmff"` is currently supported
        metal_atom_indices: metal atoms to exclude from MMFF typing
        constraint_model: geometry constraints to apply during minimization
        **kwargs: additional arguments for minimizer

    Returns:
        Minimizer instance

    Raises:
        OpenConfValueError: unknown minimizer name
    """
    if name == "rdkit_mmff":
        return RDKitMMFFMinimizer(
            metal_atom_indices=metal_atom_indices,
            constraint_model=constraint_model or ConstraintModel.empty(),
            **kwargs,
        )
    raise OpenConfValueError(f"Unknown minimizer: {name}")
