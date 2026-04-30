"""
Validation tests for the native QCDark2 SRDM dielectric engine.

Physics benchmark:
    material=Si (Si_comp.h5 composite dielectric)
    mX=48232.9466 eV (~50 keV grid point)
    sigma_e=1.098541e-38 cm^2
    FDMn=2 (light mediator)
    screening='rpa', variant='composite'
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import numericalunits as nu
import pytest

from DMeRates.data.registry import DataRegistry
from conftest import QCDARK2_SRDM_REFS


_SI_COMP_PATH = DataRegistry.qcdark2_dielectric("Si", "composite")
_FLUX_PATH = DataRegistry.srdm_flux_file("srdm_dphidv_DPLM_row10_col8.txt")
QCDARK2_DATA_AVAILABLE = _SI_COMP_PATH.is_file() and _FLUX_PATH.is_file()
_SKIP_REASON = (
    f"QCDark2 SRDM data not found. Need: {_SI_COMP_PATH} and {_FLUX_PATH}."
)

_MX_EV = 48232.9466
_SIGMA_E_CM2 = 1.098541e-38


def _compute(mode: str):
    from DMeRates.engines.dielectric import compute_dRdE

    return compute_dRdE(
        material="Si",
        mX_eV=_MX_EV,
        sigma_e_cm2=_SIGMA_E_CM2,
        FDMn=2,
        mediator_spin=mode,
        halo_model="srdm",
        screening="rpa",
        variant="composite",
    )


@pytest.mark.skipif(not QCDARK2_DATA_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("mode", ["vector", "scalar", "approx", "approx_full"])
def test_qcdark2_srdm_si_light_mediator_modes_match_references(fix_units, mode):
    """Native Si SRDM dR/dE agrees with notebook references within 5%."""
    res = _compute(mode)

    ref_key = "Si_50keV_vector_light" if mode == "vector" else f"Si_50keV_{mode}_light"
    refs = QCDARK2_SRDM_REFS[ref_key]
    for E_target, dRdE_ref in refs:
        idx = int(np.argmin(np.abs(res.E_eV - E_target)))
        actual = float(res.dRdE_per_kg_per_year_per_eV[idx])
        rel = abs(actual - dRdE_ref) / dRdE_ref
        assert rel < 0.05, (
            f"mode={mode}, E={res.E_eV[idx]:.2f} eV: "
            f"dR/dE={actual:.6e} vs ref={dRdE_ref:.6e} "
            f"(rel diff {rel*100:.4f}%) exceeds 5% tolerance."
        )


@pytest.mark.skipif(not QCDARK2_DATA_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize(
    "requested,normalized",
    [
        ("vector", "vector"),
        ("scalar", "scalar"),
        ("approx", "approx"),
        ("approx_full", "approx_full"),
        ("approx full", "approx_full"),
    ],
)
def test_qcdark2_srdm_metadata_and_aliasing(fix_units, requested, normalized):
    from DMeRates.spectrum import RateSpectrum

    res = _compute(requested)
    assert isinstance(res.spectrum, RateSpectrum)
    assert res.spectrum.backend == "qcdark2"

    meta = res.spectrum.metadata
    assert meta["halo_model"] == "srdm"
    assert meta["mediator_spin"] == normalized
    assert meta["flux_mediator_spin"] == "vector"
    assert "flux_file" in meta
    assert meta["FDMn"] == 2
    assert meta["screening"] == "rpa"

    spec_bare = res.spectrum.dR_dE.cpu().numpy() * (nu.kg * nu.year * nu.eV)
    assert np.allclose(spec_bare, res.dRdE_per_kg_per_year_per_eV, rtol=1e-12, atol=0.0)


def test_qcdark2_srdm_invalid_mediator_spin_raises():
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(ValueError, match="Supported: vector, scalar, approx, approx_full"):
        compute_dRdE(
            material="Si",
            mX_eV=_MX_EV,
            sigma_e_cm2=_SIGMA_E_CM2,
            FDMn=2,
            mediator_spin="bad_mode",
            halo_model="srdm",
            screening="rpa",
        )


def test_qcdark2_srdm_missing_manifest_entry_raises():
    """Unregistered (mX, sigma) pair raises FileNotFoundError citing the manifest."""
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(FileNotFoundError, match="manifest"):
        compute_dRdE(
            material="Si",
            mX_eV=12345.0,
            sigma_e_cm2=1e-99,
            FDMn=2,
            mediator_spin="vector",
            halo_model="srdm",
            screening="rpa",
        )


def test_qcdark2_srdm_screening_required():
    """screening=None raises ValueError for SRDM path."""
    from DMeRates.engines.dielectric import compute_dRdE

    with pytest.raises(ValueError, match="screening"):
        compute_dRdE(
            material="Si",
            mX_eV=_MX_EV,
            sigma_e_cm2=_SIGMA_E_CM2,
            FDMn=2,
            mediator_spin="vector",
            halo_model="srdm",
            screening=None,
        )


@pytest.mark.skipif(not QCDARK2_DATA_AVAILABLE, reason=_SKIP_REASON)
@pytest.mark.parametrize("mode", ["vector", "scalar", "approx", "approx_full", "approx full"])
def test_qcdark2_srdm_calculate_rates_uses_mev_public_mass(fix_units, mode):
    """Public calculate_rates keeps the legacy MeV mX_array convention for SRDM."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("Si", form_factor_type="qcdark2")
    rates = dm.calculate_rates(
        mX_array=[_MX_EV / 1.0e6],
        halo_model="srdm",
        FDMn=2,
        ne=[1],
        screening="rpa",
        sigma_e=_SIGMA_E_CM2,
        mediator_spin=mode,
    )
    assert rates.shape == (1, 1)
    assert np.isfinite(float(rates[0, 0]))
    assert float(rates[0, 0]) > 0.0
