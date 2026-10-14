"""Validate OpenConf CPU and nvMolKit execution on an allocated GPU."""

import argparse
import json
import math
import os
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

SMILES = (
    "CCCCc1ccccc1",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC1=NN(C(=C1)C2=CC=CC=C2)C3=CC=C(C=C3)S(N)(=O)=O",
)


def _require_allocated_gpu() -> None:
    """Require execution inside a Slurm job with an assigned GPU."""
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("GPU validation must run inside a Slurm job.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Slurm job has no visible GPU allocation.")


def _run_case(smiles: str, backend: str) -> dict[str, object]:
    """Generate one ensemble and return validation metrics."""
    from openconf import generate_conformers, preset_config

    config = replace(
        preset_config("rapid"),
        backend=backend,
        collect_stats=True,
        max_out=8,
        random_seed=20261014,
    )
    started = time.perf_counter()
    ensemble = generate_conformers(smiles, config=config)
    elapsed = time.perf_counter() - started
    finite_energies = [energy for energy in ensemble.energies if math.isfinite(energy)]
    if not finite_energies:
        raise RuntimeError(f"{backend} produced no finite energies for {smiles}.")
    if ensemble.generation_stats.get("compute_backend") != backend:
        raise RuntimeError(f"Expected {backend} statistics for {smiles}.")

    return {
        "backend": backend,
        "elapsed_s": elapsed,
        "finite_energy_count": len(finite_energies),
        "minimum_energy_kcal_mol": min(finite_energies),
        "n_conformers": ensemble.n_conformers,
        "smiles": smiles,
        "stats": ensemble.generation_stats,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run CPU and GPU validation cases and write a JSON report."""
    _require_allocated_gpu()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the allocated GPU.")

    args = _parse_args()
    report = {
        "cuda_device": torch.cuda.get_device_name(),
        "cuda_runtime": torch.version.cuda,
        "nvmolkit_version": version("nvmolkit"),
        "openconf_version": version("openconf"),
        "rdkit_version": version("rdkit"),
        "results": [_run_case(smiles, backend) for smiles in SMILES for backend in ("rdkit", "nvmolkit")],
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "torch_version": torch.__version__,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
