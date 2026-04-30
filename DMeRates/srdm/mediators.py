"""Mediator-spin normalization and flux-routing policy for SRDM."""

from __future__ import annotations


_SUPPORTED_MEDIATOR_SPINS: tuple[str, ...] = (
    "vector",
    "scalar",
    "approx",
    "approx_full",
)
_ALIASES: dict[str, str] = {
    "approx full": "approx_full",
}


def supported_mediator_spins() -> tuple[str, ...]:
    """Return canonical public SRDM mediator-spin names."""
    return _SUPPORTED_MEDIATOR_SPINS


def _supported_string() -> str:
    return ", ".join(supported_mediator_spins())


def normalize_mediator_spin(value) -> str:
    """Normalize public mediator-spin inputs to canonical names.

    Accepted names:
        vector, scalar, approx, approx_full
    Accepted alias:
        approx full -> approx_full
    """
    if value is None:
        candidate = "vector"
    else:
        candidate = str(value).strip().lower()
    if candidate in _ALIASES:
        candidate = _ALIASES[candidate]
    if candidate not in _SUPPORTED_MEDIATOR_SPINS:
        raise ValueError(
            f"Unsupported mediator_spin={value!r}. "
            f"Supported: {_supported_string()}."
        )
    return candidate


def flux_mediator_spin(value) -> str:
    """Return mediator-spin key used for SRDM flux-file lookup.

    Current policy intentionally reuses vector DPLM flux files for every
    detector-kernel mediator mode.
    """
    _ = normalize_mediator_spin(value)
    return "vector"

