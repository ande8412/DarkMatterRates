from pathlib import Path

import numpy as np
import pytest
import torch
import numericalunits as nu


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "srdmbeam"


def test_resolve_srdmbeam_flux_source_uses_fixture_ring():
    from DMeRates.srdm.flux_loader import resolve_srdm_flux_source

    v_over_c, dphi_dv, metadata = resolve_srdm_flux_source(
        source="SRDMBeam",
        mX_MeV=100.0,
        sigma_e_cm2=1e-36,
        FDMn=0,
        mediator_spin="vector",
        ring_index=1,
        base_data_dir=FIXTURE_ROOT,
    )

    assert v_over_c.shape == dphi_dv.shape
    assert metadata["halo_model"] == "srdm_modulated"
    assert metadata["flux_source"] == "SRDMBeam"
    assert metadata["ring_index"] == 1
    assert "isoangle_1.txt" in metadata["flux_file"]


def test_resolve_srdmbeam_flux_source_accepts_upstream_source(monkeypatch):
    from DMeRates.srdm import flux_loader

    captured = {}

    def fake_load(mX_MeV, sigma_e_cm2, FDMn, ring_index, *, modulated_source, base_data_dir):
        captured.update(
            {
                "mX_MeV": mX_MeV,
                "sigma_e_cm2": sigma_e_cm2,
                "FDMn": FDMn,
                "ring_index": ring_index,
                "modulated_source": modulated_source,
                "base_data_dir": base_data_dir,
            }
        )
        return (
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([2.0], dtype=torch.float64),
            {"halo_model": "srdm_modulated", "flux_source": "SRDMBeam"},
        )

    monkeypatch.setattr(flux_loader, "load_srdmbeam_flux", fake_load)
    _v_over_c, _dphi_dv, metadata = flux_loader.resolve_srdm_flux_source(
        source="DaMaSCUS",
        mX_MeV=100.0,
        sigma_e_cm2=1e-36,
        FDMn=0,
        mediator_spin="vector",
        ring_index=1,
        base_data_dir=FIXTURE_ROOT,
    )

    assert captured["modulated_source"] == "DaMaSCUS"
    assert metadata["flux_source"] == "SRDMBeam"
    assert metadata["mediator_spin"] == "vector"


def test_srdm_modulated_public_guards_raise_clear_errors():
    from DMeRates.DMeRate import (
        DMeRate,
        _resolve_ordinary_modulation_source,
        _validate_srdm_modulated_call,
    )

    dummy = object.__new__(DMeRate)

    with pytest.raises(ValueError, match="requires an explicit integer"):
        DMeRate.calculate_rates(dummy, 100.0, "srdm_modulated", 0, [1])

    with pytest.raises(ValueError, match="Unsupported SRDMBeam modulated_source"):
        DMeRate.calculate_rates(
            dummy,
            100.0,
            "srdm_modulated",
            0,
            [1],
            isoangle=0,
            modulated_source="Other",
        )

    assert _validate_srdm_modulated_call(
        "srdm_modulated",
        0,
        "DaMaSCUS",
    ) == "DaMaSCUS"

    assert _resolve_ordinary_modulation_source(
        "modulated",
        True,
        "DaMaSCUS",
    ) is False
    assert _resolve_ordinary_modulation_source(
        "modulated",
        False,
        "Verne",
    ) is True
    with pytest.raises(ValueError, match="For SRDMBeam use"):
        _resolve_ordinary_modulation_source("srdm_modulated", True, "Verne")

    with pytest.raises(ValueError, match="only supported with halo_model='srdm_modulated'"):
        DMeRate.calculate_rates(
            dummy,
            100.0,
            "srdm",
            0,
            [1],
            modulated_source="SRDMBeam",
        )


def test_semiconductor_route_passes_srdmbeam_ring_to_srdm_engine(monkeypatch):
    import DMeRates.DMeRate as dmr_module
    from DMeRates.DMeRate import DMeRate
    from DMeRates.spectrum import RateSpectrum

    captured = {}

    def fake_spectrum(**kwargs):
        captured.update(kwargs)
        return RateSpectrum(
            E=torch.tensor([1.0, 2.0], dtype=torch.float64) * nu.eV,
            dR_dE=torch.tensor([2.0, 2.0], dtype=torch.float64) / (nu.kg * nu.year * nu.eV),
            material="Si",
            backend="test",
            metadata={},
        )

    monkeypatch.setattr(dmr_module, "compute_form_factor_spectrum", fake_spectrum)

    dummy = object.__new__(DMeRate)
    dummy.form_factor_type = "qcdark"
    dummy.material = "Si"
    dummy.QEDark = False
    dummy.device = "cpu"
    dummy.cross_section = 1e-36 * nu.cm**2
    dummy.probabilities = torch.ones((1, 2), dtype=torch.float64)
    dummy.qArr = torch.tensor([1.0, 2.0], dtype=torch.float64)
    dummy.Earr = torch.tensor([1.0, 2.0], dtype=torch.float64) * nu.eV
    dummy.Ei_array = torch.tensor([0, 1])
    dummy.dtype_str = "float64"
    dummy.form_factor = object()

    result = DMeRate.calculate_semiconductor_rates(
        dummy,
        100.0,
        "srdm_modulated",
        0,
        [1],
        isoangle=1,
        modulated_source="SRDMBeam",
        srdm_base_data_dir=FIXTURE_ROOT,
    )

    assert result.shape == (1, 1)
    assert captured["halo_model"] == "srdm_modulated"
    assert captured["ring_index"] == 1
    assert captured["modulated_source"] == "SRDMBeam"
    assert captured["srdm_base_data_dir"] == FIXTURE_ROOT


class _FakeSRDMRates:
    def __init__(self, values_by_ring):
        self.values_by_ring = values_by_ring
        self.calls = []

    def update_crosssection(self, sigma_e):
        self.sigma_e = sigma_e

    def calculate_rates(self, _mX, halo_model, _fdm, ne, **kwargs):
        self.calls.append((halo_model, kwargs))
        value = self.values_by_ring[int(kwargs["isoangle"])]
        n = 1 if isinstance(ne, int) else len(ne)
        return torch.full((n, 1), float(value), dtype=torch.float64)


def test_identical_srdmbeam_ring_rates_give_zero_daily_modulation():
    from modulation_study.Modulation import get_srdm_daily_modulation_amplitude

    fake = _FakeSRDMRates({0: 7.0, 1: 7.0})
    summary = get_srdm_daily_modulation_amplitude(
        "Si",
        100.0,
        1e-36,
        0,
        [1],
        "SNO",
        [15, 2, 2016],
        cadence_minutes=720,
        dmRateObject=fake,
        base_data_dir=FIXTURE_ROOT,
    )

    assert np.allclose(summary["fractional_modulation"], 0.0)
    assert {call[0] for call in fake.calls} == {"srdm_modulated"}


def test_srdm_modulated_rates_use_metadata_file_angles():
    from modulation_study.Modulation import get_srdm_modulated_rates

    fake = _FakeSRDMRates({0: 10.0, 1: 20.0})
    angles, rates = get_srdm_modulated_rates(
        "Si",
        100.0,
        1e-36,
        0,
        [1],
        dmRateObject=fake,
        base_data_dir=FIXTURE_ROOT,
    )

    assert np.allclose(np.asarray(angles), np.array([0.18, 92.57]))
    assert np.allclose(np.asarray(rates).flatten(), np.array([10.0, 20.0]))


def test_srdm_daily_interpolation_uses_metadata_file_angles(monkeypatch):
    import modulation_study.isoangle as isoangle
    from modulation_study.Modulation import get_srdm_daily_rates

    def fake_angle_series(_location, _date, cadence_minutes=10):
        midpoint = 0.5 * (0.18 + 92.57)
        return "times", np.array([0.0, midpoint, 180.0])

    monkeypatch.setattr(isoangle, "solar_daily_angle_series", fake_angle_series)

    fake = _FakeSRDMRates({0: 10.0, 1: 20.0})
    _times, angles, rates = get_srdm_daily_rates(
        "Si",
        100.0,
        1e-36,
        0,
        [1],
        "SNO",
        [15, 2, 2016],
        dmRateObject=fake,
        base_data_dir=FIXTURE_ROOT,
    )

    assert np.array_equal(angles, np.array([0.0, 46.375, 180.0]))
    assert np.all(rates >= 10.0)
    assert np.all(rates <= 20.0)
    assert rates[1, 0] == pytest.approx(15.0)


def test_damascus_bin_average_representative_angles_are_bin_centers():
    from modulation_study.Modulation import _srdmbeam_representative_angles_for_indices
    from DMeRates.srdm.flux_loader import load_srdmbeam_metadata

    metadata = load_srdmbeam_metadata(
        100.0,
        1e-36,
        0,
        modulated_source="DaMaSCUS",
        base_data_dir=FIXTURE_ROOT,
    )

    angles = _srdmbeam_representative_angles_for_indices(metadata, [0, 1])

    assert np.allclose(angles, [22.5, 67.5])
    assert not np.allclose(angles, [0.0, 45.0])
