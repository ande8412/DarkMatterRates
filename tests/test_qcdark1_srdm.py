"""
SRDM tests for the QCDark1 form-factor backend.

Vector reference points are preserved from the existing SRDM derivation
notebook; additional mediator modes are finite-positive smoke checks.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import numericalunits as nu
import pytest

from DMeRates.data.registry import DataRegistry


QCDARK1_DATA_AVAILABLE = DataRegistry.qcdark1_ff("Si").exists()
_M_X_EV = 48232.9466
_SIGMA_E_CM2 = 1.098541e-38
QCDARK1_SRDM_REFS = {
    "Si_50keV_vector_light_unscreened": [
        (5.25, 6.256594e00),
        (6.95, 1.862771e01),
        (9.05, 1.368568e01),
    ],
    "Si_50keV_vector_light_screened": [
        (18.15, 2.241134e04),
        (18.35, 3.605890e03),
        (24.35, 9.339712e02),
    ],
}


def _qcdark1_srdm_spectrum(*, DoScreen, mediator_spin="vector", screening=None):
    from DMeRates.engines.form_factor import _compute_dRdE_srdm_form_factor
    from DMeRates.responses.qcdark1 import form_factor

    ff = form_factor(str(DataRegistry.qcdark1_ff("Si")))
    return _compute_dRdE_srdm_form_factor(
        backend="qcdark1",
        material="Si",
        mX_eV=_M_X_EV,
        sigma_e_cm2=_SIGMA_E_CM2,
        FDMn=2,
        mediator_spin=mediator_spin,
        DoScreen=DoScreen,
        form_factor=ff,
        screening=screening,
    )


def _assert_reference_points(spectrum, refs):
    E = (spectrum.E / nu.eV).detach().cpu().numpy()
    dRdE = (spectrum.dR_dE * nu.kg * nu.year * nu.eV).detach().cpu().numpy()
    for E_ref, rate_ref in refs:
        idx = int(np.argmin(np.abs(E - E_ref)))
        assert abs(E[idx] - E_ref) < 1e-9
        assert np.isclose(dRdE[idx], rate_ref, rtol=0.05, atol=0.0), (
            f"E={E_ref:.2f} eV changed: got {dRdE[idx]:.6e}, expected {rate_ref:.6e}"
        )


@pytest.mark.skipif(not QCDARK1_DATA_AVAILABLE, reason="QCDark1 HDF5 missing")
def test_qcdark1_srdm_si_vector_light_screened(fix_units):
    """Si QCDark1 SRDM with Thomas-Fermi screening matches notebook reference."""
    spectrum = _qcdark1_srdm_spectrum(DoScreen=True, mediator_spin="vector")
    refs = QCDARK1_SRDM_REFS["Si_50keV_vector_light_screened"]
    _assert_reference_points(spectrum, refs)
    assert spectrum.metadata["halo_model"] == "srdm"
    assert spectrum.metadata["DoScreen"] is True


@pytest.mark.skipif(not QCDARK1_DATA_AVAILABLE, reason="QCDark1 HDF5 missing")
def test_qcdark1_srdm_si_vector_light_unscreened(fix_units):
    """Si QCDark1 SRDM unscreened matches notebook reference."""
    spectrum = _qcdark1_srdm_spectrum(DoScreen=False, mediator_spin="vector")
    refs = QCDARK1_SRDM_REFS["Si_50keV_vector_light_unscreened"]
    _assert_reference_points(spectrum, refs)
    assert spectrum.backend == "qcdark1"
    assert spectrum.metadata["DoScreen"] is False


@pytest.mark.skipif(not QCDARK1_DATA_AVAILABLE, reason="QCDark1 HDF5 missing")
@pytest.mark.parametrize("mode", ["vector", "scalar", "approx", "approx_full", "approx full"])
def test_qcdark1_srdm_modes_are_finite_positive_and_metadata(mode):
    spectrum = _qcdark1_srdm_spectrum(DoScreen=True, mediator_spin=mode)
    dRdE = (spectrum.dR_dE * nu.kg * nu.year * nu.eV).detach().cpu().numpy()

    expected_mode = "approx_full" if mode == "approx full" else mode
    assert np.all(np.isfinite(dRdE))
    assert np.any(dRdE > 0.0)
    assert spectrum.metadata["mediator_spin"] == expected_mode
    assert spectrum.metadata["flux_mediator_spin"] == "vector"


def test_qcdark1_srdm_invalid_mediator_spin_raises():
    from DMeRates.engines.form_factor import _compute_dRdE_srdm_form_factor

    with pytest.raises(ValueError, match="Supported: vector, scalar, approx, approx_full"):
        _compute_dRdE_srdm_form_factor(
            backend="qcdark1",
            material="Si",
            mX_eV=_M_X_EV,
            sigma_e_cm2=_SIGMA_E_CM2,
            FDMn=2,
            mediator_spin="scalar-ish",
            DoScreen=False,
        )


def test_qcdark1_srdm_missing_manifest_entry_raises():
    from DMeRates.engines.form_factor import _compute_dRdE_srdm_form_factor

    with pytest.raises(FileNotFoundError) as exc:
        _compute_dRdE_srdm_form_factor(
            backend="qcdark1",
            material="Si",
            mX_eV=1.0e3,
            sigma_e_cm2=1.0e-40,
            FDMn=2,
            mediator_spin="vector",
            DoScreen=False,
        )
    assert "manifest" in str(exc.value).lower()


@pytest.mark.skipif(not QCDARK1_DATA_AVAILABLE, reason="QCDark1 HDF5 missing")
def test_qcdark1_srdm_calculate_rates_uses_mev_public_mass(fix_units):
    """Public calculate_rates keeps the legacy MeV mX_array convention for SRDM."""
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("Si", form_factor_type="qcdark")
    rates = dm.calculate_rates(
        mX_array=[_M_X_EV / 1.0e6],
        halo_model="srdm",
        FDMn=2,
        ne=[1],
        DoScreen=False,
        sigma_e=_SIGMA_E_CM2,
        mediator_spin="approx full",
    )
    assert rates.shape == (1, 1)
    assert np.isfinite(float(rates[0, 0]))
    assert float(rates[0, 0]) > 0.0


@pytest.mark.skipif(not QCDARK1_DATA_AVAILABLE, reason="QCDark1 HDF5 missing")
def test_qcdark1_srdm_lindhard_returns_finite_positive(fix_units):
    """Si QCDark1 SRDM supports analytic Lindhard screening."""
    spectrum = _qcdark1_srdm_spectrum(
        DoScreen=False,
        mediator_spin="vector",
        screening="lindhard",
    )
    dRdE = (spectrum.dR_dE * nu.kg * nu.year * nu.eV).detach().cpu().numpy()

    assert spectrum.metadata["screening"] == "lindhard"
    assert np.all(np.isfinite(dRdE))
    assert np.any(dRdE > 0.0)
