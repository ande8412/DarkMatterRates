"""Screening selector for legacy QEDark/QCDark1 semiconductor paths."""

from __future__ import annotations

import torch

from DMeRates.screening.lindhard import (
    DEFAULT_LINDHARD_ETA_EV,
    lindhard_screening,
    lindhard_tfscreening,
)
from DMeRates.screening.thomas_fermi import thomas_fermi_screening, tfscreening


VALID_SEMICONDUCTOR_SCREENING = {"none", "thomas_fermi", "lindhard"}
_ALIASES = {
    "off": "none",
    "false": "none",
    "no": "none",
    "tf": "thomas_fermi",
    "thomas-fermi": "thomas_fermi",
    "thomasfermi": "thomas_fermi",
}


def normalize_semiconductor_screening(screening: str | None, do_screen: bool) -> str:
    """Normalize legacy screening options while preserving DoScreen behavior."""
    if screening is None:
        return "thomas_fermi" if do_screen else "none"
    if not isinstance(screening, str):
        raise ValueError(
            f"screening={screening!r} not recognized. "
            f"Use one of: {sorted(VALID_SEMICONDUCTOR_SCREENING)}."
        )
    screening_norm = screening.strip().lower().replace(" ", "_")
    screening_norm = _ALIASES.get(screening_norm, screening_norm)
    if screening_norm not in VALID_SEMICONDUCTOR_SCREENING:
        raise ValueError(
            f"screening={screening!r} not recognized for QEDark/QCDark1. "
            f"Use one of: {sorted(VALID_SEMICONDUCTOR_SCREENING)}."
        )
    return screening_norm


def semiconductor_screening_factor(
    material: str,
    q: torch.Tensor,
    E: torch.Tensor,
    *,
    screening: str | None,
    do_screen: bool,
    lindhard_eta_eV: float = DEFAULT_LINDHARD_ETA_EV,
) -> torch.Tensor:
    """Return amplitude-level screening on a q x E grid."""
    mode = normalize_semiconductor_screening(screening, do_screen)
    if mode == "none":
        q_t = torch.as_tensor(q)
        E_t = torch.as_tensor(E)
        return torch.ones(
            (q_t.flatten().shape[0], E_t.flatten().shape[0]),
            dtype=torch.get_default_dtype(),
            device=q_t.device,
        )
    if mode == "thomas_fermi":
        return thomas_fermi_screening(material, q, E, do_screen=True)
    return lindhard_screening(
        material,
        q,
        E,
        do_screen=True,
        eta_eV=lindhard_eta_eV,
    )


def semiconductor_native_screening_factor(
    material: str,
    Earr: torch.Tensor,
    qArr: torch.Tensor,
    *,
    screening: str | None,
    do_screen: bool,
    lindhard_eta_eV: float = DEFAULT_LINDHARD_ETA_EV,
) -> torch.Tensor:
    """Return amplitude-level screening on the legacy native E x q grid."""
    mode = normalize_semiconductor_screening(screening, do_screen)
    if mode == "none":
        return torch.ones(
            (len(Earr), len(qArr)),
            dtype=torch.get_default_dtype(),
            device=qArr.device,
        )
    if mode == "thomas_fermi":
        return tfscreening(material, Earr, qArr, do_screen=True)
    return lindhard_tfscreening(
        material,
        Earr,
        qArr,
        do_screen=True,
        eta_eV=lindhard_eta_eV,
    )
