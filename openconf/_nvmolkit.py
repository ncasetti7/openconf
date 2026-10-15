"""Shared nvMolKit compatibility helpers."""

from importlib import import_module
from os import getenv
from typing import Literal

from .exceptions import OpenConfValueError


def precision_kwargs(precision: str) -> dict[str, object]:
    """Build optional nvMolKit precision arguments."""
    if precision == "full":
        return {}
    try:
        module = import_module("nvmolkit.types")
        precision_mode = module.PrecisionMode
    except (AttributeError, ImportError) as exc:
        raise OpenConfValueError("nvmolkit_precision='single' requires nvMolKit 0.7 or newer.") from exc
    return {"precision": precision_mode.SINGLE}


def _positive_environment_integer(name: str) -> int:
    """Read positive integer override or return nvMolKit default marker."""
    value = getenv(name)
    if value is None:
        return -1
    try:
        parsed = int(value)
    except ValueError as exc:
        raise OpenConfValueError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise OpenConfValueError(f"{name} must be a positive integer.")
    return parsed


def hardware_options_kwargs(operation: Literal["etkdg", "mmff"]) -> dict[str, object]:
    """Build experimental nvMolKit hardware-option arguments."""
    prefix = f"OPENCONF_NVMOLKIT_{operation.upper()}"
    batch_size = _positive_environment_integer(f"{prefix}_BATCH_SIZE")
    batches_per_gpu = _positive_environment_integer(f"{prefix}_BATCHES_PER_GPU")
    preprocessing_threads = (
        _positive_environment_integer(f"{prefix}_PREPROCESSING_THREADS") if operation == "etkdg" else -1
    )
    if batch_size == batches_per_gpu == preprocessing_threads == -1:
        return {}
    try:
        module = import_module("nvmolkit.types")
        options = module.HardwareOptions(
            preprocessingThreads=preprocessing_threads,
            batchSize=batch_size,
            batchesPerGpu=batches_per_gpu,
        )
    except (AttributeError, ImportError, TypeError) as exc:
        raise OpenConfValueError("nvMolKit hardware overrides require HardwareOptions support.") from exc
    return {"hardwareOptions": options}
