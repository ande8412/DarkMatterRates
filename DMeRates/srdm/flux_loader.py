"""SRDM flux file loader.

This module is the only place that converts km/s -> v/c and applies the
dPhi/dv unit normalisation. The returned tensors follow numericalunits conventions.
"""
import torch
import numpy as np
import numericalunits as nu
from DMeRates.data.registry import DataRegistry
from DMeRates.srdm.manifest import find_entry
from DMeRates.srdm.mediators import flux_mediator_spin, normalize_mediator_spin


def load_srdm_flux(
    mX_eV: float,
    sigma_e_cm2: float,
    FDMn: int,
    mediator_spin: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load SRDM flux for the given (mX, sigma_e, FDMn, mediator_spin) point.

    Returns:
        v_over_c : torch.Tensor shape (N_v,), dimensionless v/c. Non-positive
                   velocity rows are dropped because SRDM cross sections contain
                   a 1/v^2 prefactor.
        dphi_dv  : torch.Tensor shape (N_v,), dPhi/d(v/c) in numericalunits.
                   Divide by 1/(nu.cm**2 * nu.s) to get the dimensionless
                   integration weight for a trapezoid integral over d(v/c).

    Raises:
        ValueError: if mediator_spin is invalid.
        FileNotFoundError: if no manifest entry matches the lookup tuple.
            Message includes the full lookup tuple AND the manifest path.
    """
    mediator_spin = normalize_mediator_spin(mediator_spin)
    flux_spin = flux_mediator_spin(mediator_spin)
    entry = find_entry(mX_eV, sigma_e_cm2, FDMn, mediator_spin)
    if entry is None:
        manifest_path = DataRegistry.srdm_manifest()
        raise FileNotFoundError(
            f"No SRDM flux file registered for "
            f"(mX_eV={mX_eV}, sigma_e_cm2={sigma_e_cm2}, "
            f"FDMn={FDMn}, mediator_spin={mediator_spin!r}, "
            f"flux_mediator_spin={flux_spin!r}). "
            f"See manifest at: {manifest_path}"
        )

    flux_path = DataRegistry.srdm_flux_file(entry["filename"])
    data = np.loadtxt(str(flux_path), comments='#')
    v_kms = data[:, 0]      # km/s
    dphi_raw = data[:, 1]   # cm^-2 s^-1 (km/s)^-1

    positive_velocity = v_kms > 0.0
    if not np.any(positive_velocity):
        raise ValueError(f"SRDM flux file has no positive-velocity rows: {flux_path}")
    v_kms = v_kms[positive_velocity]
    dphi_raw = dphi_raw[positive_velocity]

    # Convert velocity km/s -> dimensionless v/c
    # c_kms_bare is the speed of light expressed in km/s as a pure float
    c_kms_bare = nu.c0 / (nu.km / nu.s)
    v_over_c_np = v_kms / c_kms_bare

    # Convert dPhi/dv [cm^-2 s^-1 (km/s)^-1] -> dPhi/d(v/c) [nu units].
    # dPhi/d(v/c) = dPhi/dv_kms * c_kms_bare. The km/s factor must be a
    # bare number here; multiplying by nu.c0 would leak randomized units into
    # the flux normalization.
    dphi_dv_nu = dphi_raw * c_kms_bare / (nu.cm**2 * nu.s)

    v_tensor = torch.tensor(v_over_c_np, dtype=torch.float64)
    dphi_tensor = torch.tensor(dphi_dv_nu, dtype=torch.float64)

    return v_tensor, dphi_tensor
