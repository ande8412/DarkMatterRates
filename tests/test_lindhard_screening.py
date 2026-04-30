import sys
sys.path.insert(0, ".")

import numpy as np
import pytest
import torch

import numericalunits as nu


@pytest.mark.parametrize("material", ["Si", "Ge"])
def test_lindhard_screening_finite_positive(material, fix_units):
    from DMeRates.screening.lindhard import lindhard_screening

    q = torch.tensor([50.0, 500.0, 5000.0]) * (nu.eV / nu.c0)
    E = torch.tensor([0.1, 5.0, 20.0]) * nu.eV
    screen = lindhard_screening(material, q, E)

    assert screen.shape == (3, 3)
    assert torch.all(torch.isfinite(screen))
    assert torch.all(screen >= 0.0)


@pytest.mark.parametrize("material", ["Si", "Ge"])
def test_lindhard_static_small_q_limit_matches_thomas_fermi(material, fix_units):
    from DMeRates.screening.lindhard import lindhard_material_parameters, lindhard_screening

    params = lindhard_material_parameters(material)
    q_eV = torch.tensor([10.0, 30.0, 100.0])
    q = q_eV * (nu.eV / nu.c0)
    E = torch.tensor([0.0]) * nu.eV

    lindhard = lindhard_screening(material, q, E, eta_eV=1e-9)[:, 0]
    thomas_fermi_limit = q_eV**2 / (q_eV**2 + params["qTF_eV"] ** 2)

    assert torch.allclose(lindhard, thomas_fermi_limit, rtol=2e-3, atol=0.0)


def test_semiconductor_screening_selector_preserves_legacy_defaults(fix_units):
    from DMeRates.screening.semiconductor import semiconductor_screening_factor
    from DMeRates.screening.thomas_fermi import thomas_fermi_screening

    q = torch.tensor([100.0, 1000.0]) * (nu.eV / nu.c0)
    E = torch.tensor([1.0, 10.0]) * nu.eV

    legacy_default = semiconductor_screening_factor(
        "Si", q, E, screening=None, do_screen=True
    )
    explicit_tf = semiconductor_screening_factor(
        "Si", q, E, screening="thomas_fermi", do_screen=False
    )
    old_helper = thomas_fermi_screening("Si", q, E, do_screen=True)
    explicit_none = semiconductor_screening_factor(
        "Si", q, E, screening="none", do_screen=True
    )

    assert torch.allclose(legacy_default, old_helper)
    assert torch.allclose(explicit_tf, old_helper)
    assert torch.all(explicit_none == 1.0)


@pytest.mark.parametrize("form_factor_type", ["qedark", "qcdark"])
def test_legacy_halo_lindhard_smoke_runs(form_factor_type, fix_units):
    from DMeRates.DMeRate import DMeRate

    dm = DMeRate("Si", form_factor_type=form_factor_type)
    if form_factor_type == "qedark":
        dm.change_to_step()

    rates = dm.calculate_rates(
        mX_array=[10],
        halo_model="imb",
        FDMn=0,
        ne=[1, 2],
        DoScreen=False,
        screening="lindhard",
        integrate=False,
    )

    rates_np = rates.detach().cpu().numpy()
    assert rates.shape == (2, 1)
    assert np.all(np.isfinite(rates_np))
    assert np.any(rates_np > 0.0)


def test_qcdark2_lindhard_screening_rejected():
    from DMeRates.screening.dielectric import normalize_dielectric_screening

    with pytest.raises(ValueError, match="screening='lindhard' not recognized"):
        normalize_dielectric_screening("lindhard")
