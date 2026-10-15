"""Shared nvMolKit compatibility helpers."""

from importlib import import_module

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
