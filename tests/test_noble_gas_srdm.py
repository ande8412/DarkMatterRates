"""SRDM smoke tests for the noble-gas (wimprates) backend."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pytest
import torch

import numericalunits as nu
from DMeRates.DMeRate import DMeRate
from DMeRates.engines.noble_gas import noble_srdm_dRdE_spectrum


_M_X_EV = 48232.9466
_M_X_MEV = _M_X_EV / 1.0e6
_SIGMA_E_CM2 = 1.098541e-38


def _srdm_spectrum(dm, *, material):
    return noble_srdm_dRdE_spectrum(
        material=material,
        mX=_M_X_MEV,
        FDMn=2,
        mediator_spin="vector",
        sigma_e_cm2=_SIGMA_E_CM2,
        form_factor=dm.form_factor,
        qArrdict=dm.qArrdict,
        Earr=dm.Earr,
        reduced_mass_fn=dm.reduced_mass,
        fdm_fn=dm.FDM,
        vmin_tensor_fn=dm.vMin_tensor,
    )


@pytest.mark.parametrize("material", ["Xe", "Ar"])
def test_noble_gas_srdm_spectrum_is_finite_positive(material, fix_units):
    """Xe/Ar SRDM produces finite positive shell-summed spectra."""
    dm = DMeRate(material, form_factor_type="wimprates")
    spectrum = _srdm_spectrum(dm, material=material)
    dRdE = (spectrum.dR_dE * nu.kg * nu.year * nu.eV).detach().cpu().numpy()

    assert spectrum.backend == "noble_gas"
    assert spectrum.metadata["halo_model"] == "srdm"
    assert spectrum.metadata["mediator_spin"] == "vector"
    assert spectrum.metadata["FDMn"] == 2
    assert spectrum.metadata["mX_eV"] == pytest.approx(_M_X_EV)
    assert spectrum.metadata["sigma_e_cm2"] == pytest.approx(_SIGMA_E_CM2)
    assert "manifest" not in spectrum.metadata["flux_file"].lower()
    assert set(spectrum.shell_labels) == set(spectrum.shell_spectra)
    assert np.all(np.isfinite(dRdE))
    assert np.any(dRdE > 0.0)


def test_noble_gas_srdm_calculate_rates_uses_mev_public_mass(fix_units):
    """Public calculate_rates keeps the MeV mX_array convention for noble SRDM."""
    dm = DMeRate("Xe", form_factor_type="wimprates")
    rates = dm.calculate_rates(
        mX_array=[_M_X_MEV],
        halo_model="srdm",
        FDMn=2,
        ne=[1, 2, 3],
        sigma_e=_SIGMA_E_CM2,
    )

    assert rates.shape == (3, 1)
    assert torch.all(torch.isfinite(rates))
    assert torch.any(rates > 0.0)


def test_noble_gas_srdm_return_shells_shape(fix_units):
    """returnShells remains available for noble SRDM."""
    dm = DMeRate("Xe", form_factor_type="wimprates")
    rates, shells = dm.calculate_nobleGas_rates(
        mX_array=[_M_X_MEV],
        halo_model="srdm",
        FDMn=2,
        ne=[1, 2],
        sigma_e=_SIGMA_E_CM2,
        returnShells=True,
    )

    assert shells[0] == "Summed"
    assert rates.shape == (1, 2, len(shells))
    assert torch.all(torch.isfinite(rates))


def test_noble_gas_srdm_missing_manifest_entry_raises(fix_units):
    dm = DMeRate("Xe", form_factor_type="wimprates")
    with pytest.raises(FileNotFoundError, match="manifest"):
        dm.calculate_rates(
            mX_array=[0.001],
            halo_model="srdm",
            FDMn=2,
            ne=[1],
            sigma_e=1.0e-40,
        )


def test_noble_gas_srdm_non_vector_mediator_raises(fix_units):
    dm = DMeRate("Xe", form_factor_type="wimprates")
    with pytest.raises(NotImplementedError, match="vector"):
        dm.calculate_rates(
            mX_array=[_M_X_MEV],
            halo_model="srdm",
            FDMn=2,
            ne=[1],
            sigma_e=_SIGMA_E_CM2,
            mediator_spin="scalar",
        )
