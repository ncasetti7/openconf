"""Cross-molecule batching for conformer generation."""

import time
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem

from ..config import ConformerConfig
from ..dedupe import rmsd_dedupe
from ..embedding import get_embedder
from ..exceptions import OpenConfValueError
from ..perceive import RotorModel
from ..pool import ConformerPool
from ..relax import minimize_molecules_mmff
from ..torsionlib import TorsionLibrary, get_default_torsion_library
from .candidates import CandidateBatchWorkspace
from .hybrid import HybridProposer, _resolve_runtime_tuned_config
from .seeding import (
    PrecomputedSeeds,
    SeedEmbeddingProfile,
    populate_seed_plan_stats,
    resolve_seed_plan,
    seed_embedding_profile,
)
from .stats import GenerationStat, new_generation_stats, populate_effective_config_stats

type GenerationResult = tuple[Chem.Mol, list[int], list[float], dict[str, GenerationStat]]


@dataclass(frozen=True, slots=True)
class BatchSeedRequest:
    """Molecule prepared for cross-molecule seed batching."""

    index: int
    mol: Chem.Mol
    rotor_model: RotorModel
    config: ConformerConfig
    low_flex: bool


@dataclass(slots=True)
class _EmbeddedSeedRequest:
    """Batch request with resolved settings and embedded conformers."""

    request: BatchSeedRequest
    effective_config: ConformerConfig
    conf_ids: list[int]
    embedding_time_s: float
    embedding_batch_size: int


@dataclass(slots=True)
class _ProposalRun:
    """Mutable loop state for one molecule in proposal batching."""

    request: BatchSeedRequest
    config: ConformerConfig
    proposer: HybridProposer
    pool: ConformerPool
    stats: dict[str, GenerationStat]
    seeds: PrecomputedSeeds
    total_start: float
    step: int = 0
    stagnation: int = 0
    complete: bool = False


@dataclass(frozen=True, slots=True)
class _StagedProposal:
    """Proposal workspace ready for multi-molecule MMFF minimization."""

    run: _ProposalRun
    workspace: CandidateBatchWorkspace
    mmff_props: object


def _effective_config(request: BatchSeedRequest) -> ConformerConfig:
    """Resolve configuration used by request generation path."""
    if request.low_flex:
        return request.config
    return _resolve_runtime_tuned_config(request.config, request.rotor_model)[0]


def _embed_group(
    requests: list[BatchSeedRequest],
    profile: SeedEmbeddingProfile,
) -> tuple[list[list[int]], float, int]:
    """Embed one compatible molecule group with indexed failure isolation."""
    embedder = get_embedder("nvmolkit", precision=profile.nvmolkit_precision)
    start = time.perf_counter()
    batch_size = len(requests)
    try:
        conf_ids = embedder.embed_many(
            [request.mol for request in requests],
            profile.n_seeds,
            profile.parameters(),
        )
    except (RuntimeError, ValueError):
        batch_size = 1
        conf_ids = []
        for request in requests:
            try:
                conf_ids.append(embedder.embed(request.mol, profile.n_seeds, profile.parameters()))
            except (RuntimeError, ValueError) as exc:
                raise OpenConfValueError(
                    f"Molecule at batch index {request.index} failed during embedding: {exc}"
                ) from exc
    return conf_ids, time.perf_counter() - start, batch_size


def generate_batched_seeds(
    requests: list[BatchSeedRequest],
    torsion_library: TorsionLibrary | None = None,
) -> dict[int, PrecomputedSeeds]:
    """Embed and minimize seeds across compatible molecules.

    Args:
        requests: prepared molecules eligible for native batching
        torsion_library: torsion library override

    Returns:
        Precomputed seeds keyed by original batch index
    """
    if not requests:
        return {}

    grouped: dict[SeedEmbeddingProfile, list[BatchSeedRequest]] = defaultdict(list)
    effective_configs: dict[int, ConformerConfig] = {}
    for request in requests:
        effective_config = _effective_config(request)
        effective_configs[request.index] = effective_config
        seed_plan = resolve_seed_plan(request.mol, request.rotor_model, effective_config)
        grouped[seed_embedding_profile(request.rotor_model, effective_config, seed_plan)].append(request)

    embedded: list[_EmbeddedSeedRequest] = []
    for profile, group in grouped.items():
        conf_ids_by_mol, elapsed, batch_size = _embed_group(group, profile)
        embedded.extend(
            _EmbeddedSeedRequest(request, effective_configs[request.index], conf_ids, elapsed, batch_size)
            for request, conf_ids in zip(group, conf_ids_by_mol, strict=True)
        )

    library = torsion_library if torsion_library is not None else get_default_torsion_library()
    proposers = {
        item.request.index: HybridProposer(
            item.request.mol,
            item.request.rotor_model,
            library,
            item.effective_config,
        )
        for item in embedded
    }
    results: dict[int, PrecomputedSeeds] = {}
    minimization_groups: dict[tuple[int, str, str], list[_EmbeddedSeedRequest]] = defaultdict(list)

    for item in embedded:
        if not item.conf_ids:
            results[item.request.index] = PrecomputedSeeds(
                [],
                item.embedding_time_s,
                0.0,
                item.embedding_batch_size,
                1,
            )
            continue
        proposer = proposers[item.request.index]
        if proposer.seed_mmff_properties() is None:
            start = time.perf_counter()
            values = proposer.minimize_embedded_seeds(item.conf_ids)
            results[item.request.index] = PrecomputedSeeds(
                values,
                item.embedding_time_s,
                time.perf_counter() - start,
                item.embedding_batch_size,
                1,
            )
        else:
            minimization_groups[
                (
                    item.effective_config.seed_minimization_iters,
                    item.effective_config.nvmolkit_minimizer,
                    item.effective_config.nvmolkit_precision,
                )
            ].append(item)

    for (max_iters, minimizer_kind, precision), group in minimization_groups.items():
        group_proposers = [proposers[item.request.index] for item in group]
        properties = [proposer.seed_mmff_properties() for proposer in group_proposers]
        if any(props is None for props in properties):
            raise OpenConfValueError("MMFF properties became unavailable during batch seed minimization.")
        start = time.perf_counter()
        try:
            energies_by_mol = minimize_molecules_mmff(
                [item.request.mol for item in group],
                properties,
                [item.conf_ids for item in group],
                max_iters,
                backend="nvmolkit",
                minimizer_kind=minimizer_kind,
                precision=precision,
            )
        except (RuntimeError, ValueError):
            energies_by_mol = []
            for item, proposer in zip(group, group_proposers, strict=True):
                try:
                    values = proposer.minimize_embedded_seeds(item.conf_ids)
                except (RuntimeError, ValueError) as exc:
                    raise OpenConfValueError(
                        f"Molecule at batch index {item.request.index} failed during minimization: {exc}"
                    ) from exc
                results[item.request.index] = PrecomputedSeeds(
                    values,
                    item.embedding_time_s,
                    time.perf_counter() - start,
                    item.embedding_batch_size,
                    1,
                )
            continue

        elapsed = time.perf_counter() - start
        for item, proposer, energies in zip(group, group_proposers, energies_by_mol, strict=True):
            seed_results = list(zip(item.conf_ids, energies, strict=True))
            values = proposer.complete_preminimized_seeds(seed_results)
            results[item.request.index] = PrecomputedSeeds(
                values,
                item.embedding_time_s,
                elapsed,
                item.embedding_batch_size,
                len(group),
            )

    return results


def _create_proposal_run(
    request: BatchSeedRequest,
    seeds: PrecomputedSeeds,
    torsion_library: TorsionLibrary,
) -> _ProposalRun:
    """Initialize proposal-loop state from precomputed seeds."""
    total_start = time.perf_counter()
    config, tuned_defaults_applied = _resolve_runtime_tuned_config(request.config, request.rotor_model)
    seed_plan = resolve_seed_plan(request.mol, request.rotor_model, config)
    stats = new_generation_stats() if config.collect_stats else {}
    populate_effective_config_stats(
        stats,
        config=config,
        tuned_defaults_applied=tuned_defaults_applied,
        seed_prune_rms_thresh=seed_plan.prune_rms_thresh,
    )
    populate_seed_plan_stats(stats, seed_plan)
    if stats:
        stats["seed_embedding_time_s"] = seeds.embedding_time_s
        stats["seed_minimization_time_s"] = seeds.minimization_time_s
        stats["seed_embedding_batch_size"] = seeds.embedding_batch_size
        stats["seed_minimization_batch_size"] = seeds.minimization_batch_size
        stats["seed_time_s"] = seeds.embedding_time_s + seeds.minimization_time_s
        stats["n_seed_conformers"] = len(seeds.values)

    proposer = HybridProposer(
        request.mol,
        request.rotor_model,
        torsion_library,
        config,
        stats=stats or None,
    )
    proposer.prepare_minimizers()
    pool = ConformerPool(request.mol, config)
    for conf_id, energy in seeds.values:
        pool.insert(conf_id, energy, source="seed_etkdg")
    return _ProposalRun(
        request,
        config,
        proposer,
        pool,
        stats,
        seeds,
        total_start,
        complete=config.n_steps <= 0,
    )


def _advance_proposal_run(
    run: _ProposalRun,
    results: list[tuple[int, float, str]],
) -> None:
    """Commit one completed proposal round and update loop state."""
    step_increment = run.config.minimize_batch_size
    run.step += step_increment
    any_accepted = run.proposer.accept_proposals(run.pool, results)
    run.stagnation = 0 if any_accepted else run.stagnation + step_increment

    if run.config.patience > 0 and run.stagnation >= run.config.patience:
        run.complete = True
        return

    if run.pool.should_dedupe():
        dedupe_start = time.perf_counter()
        removed = run.pool.dedupe()
        if run.stats:
            run.stats["dedupe_time_s"] = float(run.stats["dedupe_time_s"]) + (time.perf_counter() - dedupe_start)
            run.stats["n_dedupe_calls"] = int(run.stats["n_dedupe_calls"]) + 1
            run.stats["n_dedupe_removed"] = int(run.stats["n_dedupe_removed"]) + removed
        run.proposer.record_dedupe_outcome(set(run.pool.conf_ids))

    run.complete = run.step >= run.config.n_steps


def _minimize_staged_proposals(staged: list[_StagedProposal]) -> None:
    """Minimize and commit proposal workspaces in native multi-molecule calls."""
    grouped: dict[tuple[int, str, str], list[_StagedProposal]] = defaultdict(list)
    for item in staged:
        grouped[
            (
                int(item.run.proposer.fast_minimizer.max_iters),
                item.run.config.nvmolkit_minimizer,
                item.run.config.nvmolkit_precision,
            )
        ].append(item)

    for (max_iters, minimizer_kind, precision), group in grouped.items():
        minimize_start = time.perf_counter()
        try:
            energies_by_mol = minimize_molecules_mmff(
                [item.workspace.staging_mol for item in group],
                [item.mmff_props for item in group],
                [item.workspace.stage_ids for item in group],
                max_iters,
                backend="nvmolkit",
                minimizer_kind=minimizer_kind,
                precision=precision,
            )
        except (RuntimeError, ValueError):
            for item in group:
                item_start = time.perf_counter()
                try:
                    energies = item.run.proposer.minimize_proposal_batch(item.workspace)
                except (RuntimeError, ValueError) as exc:
                    raise OpenConfValueError(
                        f"Molecule at batch index {item.run.request.index} failed during proposal minimization: {exc}"
                    ) from exc
                item.run.proposer.record_proposal_minimization(time.perf_counter() - item_start)
                results = item.run.proposer.commit_proposal_batch(item.workspace, energies)
                _advance_proposal_run(item.run, results)
            continue

        elapsed = time.perf_counter() - minimize_start
        for item, energies in zip(group, energies_by_mol, strict=True):
            item.run.proposer.record_proposal_minimization(elapsed, len(group))
            results = item.run.proposer.commit_proposal_batch(item.workspace, energies)
            _advance_proposal_run(item.run, results)


def _finalize_proposal_run(run: _ProposalRun) -> GenerationResult:
    """Select, refine, and return final conformers for one completed run."""
    selection_start = time.perf_counter()
    final_ids = run.pool.select_final()
    if run.stats:
        run.stats["final_selection_time_s"] = time.perf_counter() - selection_start
        run.stats["n_final_selected"] = len(final_ids)
        run.stats["n_steps_executed"] = run.step

    if run.config.do_final_refine:
        refine_start = time.perf_counter()
        final_energies = run.proposer.full_refine_final(
            run.request.mol,
            final_ids,
            run.config.num_threads,
            run.config.max_minimization_iters,
            dielectric=run.config.final_dielectric,
        )
        if run.stats:
            run.stats["final_refine_time_s"] = time.perf_counter() - refine_start
        energy_map = dict(zip(final_ids, final_energies, strict=True))
        if run.config.dedupe_rmsd_threshold is not None:
            final_ids = rmsd_dedupe(
                run.request.mol,
                final_ids,
                final_energies,
                use_heavy_atoms_only=run.config.use_heavy_atoms_only,
                rmsd_threshold=run.config.dedupe_rmsd_threshold,
                max_atom_deviation=run.config.dedupe_max_atom_deviation,
            )
            final_energies = [energy_map[conf_id] for conf_id in final_ids]
    else:
        energy_map = {
            conf_id: record.energy_kcal if record.energy_kcal is not None else float("inf")
            for conf_id, record in run.pool.records.items()
        }
        final_energies = [energy_map.get(conf_id, float("inf")) for conf_id in final_ids]

    final_set = set(final_ids)
    for conf_id in set(run.pool.conf_ids) - final_set:
        run.request.mol.RemoveConformer(conf_id)
    if run.stats:
        run.stats["total_time_s"] = (
            time.perf_counter() - run.total_start + run.seeds.embedding_time_s + run.seeds.minimization_time_s
        )
    return run.request.mol, final_ids, final_energies, run.stats


def generate_batched_proposals(
    requests: list[BatchSeedRequest],
    seeds: dict[int, PrecomputedSeeds],
    torsion_library: TorsionLibrary | None = None,
) -> dict[int, GenerationResult]:
    """Run hybrid proposal loops with cross-molecule MMFF batching.

    Args:
        requests: prepared unconstrained hybrid-generation inputs
        seeds: precomputed seeds keyed by original batch index
        torsion_library: torsion library override

    Returns:
        Generation results keyed by original batch index
    """
    if not requests:
        return {}
    library = torsion_library if torsion_library is not None else get_default_torsion_library()
    runs: list[_ProposalRun] = []
    for request in requests:
        try:
            runs.append(_create_proposal_run(request, seeds[request.index], library))
        except (KeyError, RuntimeError, ValueError) as exc:
            raise OpenConfValueError(
                f"Molecule at batch index {request.index} failed during proposal setup: {exc}"
            ) from exc

    while active := [run for run in runs if not run.complete]:
        staged: list[_StagedProposal] = []
        for run in active:
            proposal_start = time.perf_counter()
            try:
                workspace = run.proposer.stage_proposal_batch(run.pool, run.step)
            except (RuntimeError, ValueError) as exc:
                raise OpenConfValueError(
                    f"Molecule at batch index {run.request.index} failed during proposal staging: {exc}"
                ) from exc
            if run.stats:
                run.stats["proposal_stage_time_s"] = float(run.stats["proposal_stage_time_s"]) + (
                    time.perf_counter() - proposal_start
                )
                run.stats["n_batches"] = int(run.stats["n_batches"]) + 1
            if workspace is None:
                _advance_proposal_run(run, [])
                continue
            mmff_props = run.proposer.proposal_mmff_properties()
            if mmff_props is None:
                minimize_start = time.perf_counter()
                energies = run.proposer.minimize_proposal_batch(workspace)
                run.proposer.record_proposal_minimization(time.perf_counter() - minimize_start)
                results = run.proposer.commit_proposal_batch(workspace, energies)
                _advance_proposal_run(run, results)
                continue
            staged.append(_StagedProposal(run, workspace, mmff_props))
        _minimize_staged_proposals(staged)

    results: dict[int, GenerationResult] = {}
    for run in runs:
        try:
            results[run.request.index] = _finalize_proposal_run(run)
        except (RuntimeError, ValueError) as exc:
            raise OpenConfValueError(
                f"Molecule at batch index {run.request.index} failed during proposal finalization: {exc}"
            ) from exc
    return results
