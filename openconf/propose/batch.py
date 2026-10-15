"""Cross-molecule seed batching for conformer generation."""

import time
from collections import defaultdict
from dataclasses import dataclass

from rdkit import Chem

from ..config import ConformerConfig
from ..embedding import get_embedder
from ..exceptions import OpenConfValueError
from ..perceive import RotorModel
from ..relax import minimize_molecules_mmff
from ..torsionlib import TorsionLibrary, get_default_torsion_library
from .hybrid import HybridProposer, _resolve_runtime_tuned_config
from .seeding import PrecomputedSeeds, SeedEmbeddingProfile, resolve_seed_plan, seed_embedding_profile


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
    embedder = get_embedder("nvmolkit")
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
    minimization_groups: dict[int, list[_EmbeddedSeedRequest]] = defaultdict(list)

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
            minimization_groups[item.effective_config.seed_minimization_iters].append(item)

    for max_iters, group in minimization_groups.items():
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
