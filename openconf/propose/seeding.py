"""Seed-planning policy for conformer generation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rdkit.Chem import AllChem, rdDistGeom

from ..tuning import get_runtime_tuning

if TYPE_CHECKING:
    from rdkit import Chem

    from ..config import ConformerConfig
    from ..perceive import RotorModel
    from .stats import GenerationStat

type Config = ConformerConfig
type GenerationStatsValue = GenerationStat
type Molecule = Chem.Mol
type Model = RotorModel


def _is_large_flexible_non_macrocyclic(config: Config, rotor_model: Model) -> bool:
    """Return whether molecule matches large-flexible tuning regime."""
    tuning = get_runtime_tuning()
    return (
        config.constraint_spec is None
        and not rotor_model.ring_info.get("has_macrocycle")
        and rotor_model.n_rotatable >= tuning.large_flexible.rotatable_threshold
    )


def _count_hetero_heavy_atoms(mol: Molecule) -> int:
    """Count heavy atoms that are not carbon."""
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})


def _resolve_seed_prune_rms_thresh(
    mol: Molecule,
    rotor_model: Model,
    config: Config,
) -> float:
    """Resolve ETKDG prune threshold for current molecule."""
    tuning = get_runtime_tuning()
    threshold = config.seed_prune_rms_thresh
    if rotor_model.ring_info.get("has_macrocycle"):
        return -1.0
    if threshold < 0.0 or not bool(config.topology_aware_seed_pruning):
        return threshold
    if not _is_large_flexible_non_macrocyclic(config, rotor_model):
        return threshold

    hetero_heavy = _count_hetero_heavy_atoms(mol)
    ring_sizes = rotor_model.ring_info["ring_sizes"]
    if not ring_sizes and hetero_heavy <= 1:
        return max(threshold, tuning.large_flexible.prune_thresholds.hydrocarbon)
    if not ring_sizes and hetero_heavy <= 3:
        return max(threshold, tuning.large_flexible.prune_thresholds.acyclic)
    return max(threshold, tuning.large_flexible.prune_thresholds.flexible)


def _resolve_seed_budget_scale(mol: Molecule, rotor_model: Model, config: Config) -> float:
    """Resolve topology-aware scale factor for ETKDG seed count."""
    tuning = get_runtime_tuning()
    if not bool(config.topology_aware_seed_budget):
        return 1.0
    if not _is_large_flexible_non_macrocyclic(config, rotor_model):
        return 1.0

    hetero_heavy = _count_hetero_heavy_atoms(mol)
    ring_sizes = rotor_model.ring_info["ring_sizes"]
    if not ring_sizes and hetero_heavy <= 1:
        return tuning.large_flexible.seed_scales.hydrocarbon
    if not ring_sizes and hetero_heavy <= 3:
        return tuning.large_flexible.seed_scales.acyclic
    return tuning.large_flexible.seed_scales.flexible


def _resolve_seed_budget_floor(mol: Molecule, rotor_model: Model, config: Config) -> int:
    """Resolve topology-aware lower bound for auto-computed seed count."""
    tuning = get_runtime_tuning()
    default_floor = 20
    if not bool(config.topology_aware_seed_budget):
        return default_floor
    if not _is_large_flexible_non_macrocyclic(config, rotor_model):
        return default_floor

    hetero_heavy = _count_hetero_heavy_atoms(mol)
    ring_sizes = rotor_model.ring_info["ring_sizes"]
    if not ring_sizes and hetero_heavy <= 1:
        return tuning.large_flexible.seed_floors.hydrocarbon
    if not ring_sizes and hetero_heavy <= 3:
        return tuning.large_flexible.seed_floors.acyclic
    return tuning.large_flexible.seed_floors.flexible


def _is_low_flex_seed_budget_candidate(
    mol: Molecule,
    rotor_model: Model,
    config: Config,
) -> bool:
    """Return whether reduced low-flexibility auto-seeding can be applied."""
    tuning = get_runtime_tuning().low_flex_path
    budget = tuning.seed_budget
    return (
        budget.enabled
        and config.constraint_spec is None
        and rotor_model.n_rotatable <= tuning.max_rotatable
        and mol.GetNumHeavyAtoms() <= budget.max_heavy_atoms
        and (tuning.allow_macrocycles or not rotor_model.ring_info.get("has_macrocycle"))
        and (tuning.allow_rings or not rotor_model.ring_info.get("ring_sizes"))
    )


_SEED_ROTOR_KINK = 8  # rotors beyond this threshold contribute n_per_rotor-1 seeds each
_MACRO_SEED_CAP = 12  # ring sizes beyond this cap grow linearly rather than quadratically


def _compute_n_seeds(rotor_model: Model, n_per_rotor: int = 3) -> int:
    """Compute topology-derived seed count before runtime reductions.

    Args:
        rotor_model: rotor model for molecule
        n_per_rotor: seeds per rotatable bond

    Returns:
        Recommended seed conformer count
    """
    n_rot = rotor_model.n_rotatable
    if n_rot <= _SEED_ROTOR_KINK:
        base = max(20, n_rot * n_per_rotor)
    else:
        tail_rate = max(1, n_per_rotor - 1)
        base = max(20, _SEED_ROTOR_KINK * n_per_rotor + (n_rot - _SEED_ROTOR_KINK) * tail_rate)
    ring_bonus = len(rotor_model.ring_flips) * 5
    macro_bonus = sum(s * min(s, _MACRO_SEED_CAP) for s in rotor_model.ring_info["ring_sizes"] if s >= 10)
    return min(500, base + ring_bonus + macro_bonus)


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Resolved ETKDG seed budget and pruning policy."""

    n_seeds: int
    base_n_seeds: int
    budget_scale: float
    budget_floor: int
    prune_rms_thresh: float
    reason: str


@dataclass(frozen=True, slots=True)
class SeedEmbeddingProfile:
    """Settings that must match within one native embedding call."""

    n_seeds: int
    random_seed: int
    num_threads: int
    prune_rms_thresh: float
    use_small_ring_torsions: bool
    use_macrocycle_torsions: bool
    use_basic_knowledge: bool
    nvmolkit_precision: str

    def parameters(self) -> rdDistGeom.EmbedParameters:
        """Build RDKit embedding parameters for this profile."""
        params = AllChem.ETKDGv3()
        params.randomSeed = self.random_seed
        params.numThreads = self.num_threads
        params.pruneRmsThresh = self.prune_rms_thresh
        params.useSmallRingTorsions = self.use_small_ring_torsions
        params.useMacrocycleTorsions = self.use_macrocycle_torsions
        params.useBasicKnowledge = self.use_basic_knowledge
        return params


@dataclass(frozen=True, slots=True)
class PrecomputedSeeds:
    """Seed results produced before a generation runner starts."""

    values: list[tuple[int, float]]
    embedding_time_s: float = 0.0
    minimization_time_s: float = 0.0
    embedding_batch_size: int = 1
    minimization_batch_size: int = 1


def seed_embedding_profile(
    rotor_model: Model,
    config: Config,
    seed_plan: SeedPlan,
) -> SeedEmbeddingProfile:
    """Resolve settings used to group molecules for seed embedding.

    Args:
        rotor_model: rotor model for molecule
        config: generation configuration
        seed_plan: resolved seed policy

    Returns:
        Hashable native embedding profile
    """
    ring_info = rotor_model.ring_info
    macrocycle_tuning = get_runtime_tuning().macrocycle_seeding
    has_macrocycle = bool(ring_info.get("has_macrocycle"))
    prune_rms_thresh = seed_plan.prune_rms_thresh
    if has_macrocycle and macrocycle_tuning.disable_prune_rms:
        prune_rms_thresh = -1.0
    return SeedEmbeddingProfile(
        n_seeds=seed_plan.n_seeds,
        random_seed=config.random_seed or -1,
        num_threads=int(config.num_threads or 0),
        prune_rms_thresh=prune_rms_thresh,
        use_small_ring_torsions=bool(ring_info.get("has_small_ring")),
        use_macrocycle_torsions=has_macrocycle and macrocycle_tuning.use_macrocycle_torsions,
        use_basic_knowledge=has_macrocycle and macrocycle_tuning.use_basic_knowledge,
        nvmolkit_precision=config.nvmolkit_precision,
    )


def resolve_seed_plan(mol: Molecule, rotor_model: Model, config: Config) -> SeedPlan:
    """Resolve seed count and pruning policy for generation run.

    Args:
        mol: molecule after preparation
        rotor_model: rotor model for `mol`
        config: generation configuration

    Returns:
        Seed planning result consumed by generation paths
    """
    prune_rms_thresh = _resolve_seed_prune_rms_thresh(mol, rotor_model, config)
    if config.n_seeds is not None:
        return SeedPlan(
            n_seeds=config.n_seeds,
            base_n_seeds=config.n_seeds,
            budget_scale=1.0,
            budget_floor=config.n_seeds,
            prune_rms_thresh=prune_rms_thresh,
            reason="explicit",
        )

    base_n_seeds = _compute_n_seeds(rotor_model, config.seed_n_per_rotor)
    budget_scale = _resolve_seed_budget_scale(mol, rotor_model, config)
    budget_floor = _resolve_seed_budget_floor(mol, rotor_model, config)
    reason = "auto"

    if _is_low_flex_seed_budget_candidate(mol, rotor_model, config):
        budget = get_runtime_tuning().low_flex_path.seed_budget
        budget_floor = budget.seed_floor
        capped_n_seeds = max(budget.seed_floor, rotor_model.n_rotatable * budget.seed_per_rotor)
        capped_n_seeds = min(capped_n_seeds, budget.max_seeds)
        return SeedPlan(
            n_seeds=min(base_n_seeds, capped_n_seeds),
            base_n_seeds=base_n_seeds,
            budget_scale=1.0,
            budget_floor=budget_floor,
            prune_rms_thresh=prune_rms_thresh,
            reason="low_flex_acyclic",
        )

    n_seeds = base_n_seeds
    if budget_scale != 1.0:
        n_seeds = max(budget_floor, round(base_n_seeds * budget_scale))
        reason = "large_flexible"

    return SeedPlan(
        n_seeds=n_seeds,
        base_n_seeds=base_n_seeds,
        budget_scale=budget_scale,
        budget_floor=budget_floor,
        prune_rms_thresh=prune_rms_thresh,
        reason=reason,
    )


def populate_seed_plan_stats(stats: dict[str, GenerationStatsValue], seed_plan: SeedPlan) -> None:
    """Populate seed-plan stats when instrumentation is enabled."""
    if not stats:
        return
    stats["seed_plan_reason"] = seed_plan.reason
    stats["seed_plan_base_n_seeds"] = seed_plan.base_n_seeds
    stats["requested_n_seeds"] = seed_plan.n_seeds
    stats["effective_seed_budget_scale"] = seed_plan.budget_scale
    stats["effective_seed_budget_floor"] = seed_plan.budget_floor
