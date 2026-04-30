"""Tests for SRDM Phase A infrastructure (flux loader + kinematics).

Note on manifest key values: the Emken-Essig-Xu (2024) flux files are on a
logspaced grid that does not hit round numbers. The two benchmark files are:
  - DPLM_10_8:  mX=48232.9466 eV (~50 keV nominal), sigma=1.098541e-38 cm^2
  - DPLM_15_20: mX=510927.744 eV (~0.5 MeV nominal), sigma=1.151395e-37 cm^2
The manifest records actual grid values; smoke tests use those exact values.
"""
import math
import numpy as np
import pytest
import torch
import numericalunits as nu


class TestGamma:
    def test_gamma_at_zero(self):
        """gamma(0) == 1 exactly."""
        v = torch.tensor(0.0, dtype=torch.float64)
        from DMeRates.srdm.kinematics import gamma
        assert float(gamma(v)) == pytest.approx(1.0, rel=1e-12)

    def test_gamma_at_half(self):
        """gamma(0.5) ~ 1.15470."""
        v = torch.tensor(0.5, dtype=torch.float64)
        from DMeRates.srdm.kinematics import gamma
        expected = 1.0 / math.sqrt(1 - 0.25)
        assert float(gamma(v)) == pytest.approx(expected, rel=1e-8)


class TestVmin:
    def test_v_min_recovers_halo_in_gamma_one(self):
        """In v << 1 (gamma ~ 1), v_min_relativistic reduces to q/(2mX) + omega/q."""
        from DMeRates.srdm.kinematics import v_min_relativistic, gamma

        mX_eV = 1e6
        q_eV = torch.tensor([10.0, 100.0, 500.0], dtype=torch.float64)
        omega_eV = torch.tensor([1.0, 5.0, 20.0], dtype=torch.float64)
        v = torch.tensor(1e-4, dtype=torch.float64)
        gam = gamma(v)

        vmin_rel = v_min_relativistic(q_eV[:, None], omega_eV[None, :], mX_eV, gam)
        vmin_halo = q_eV[:, None] / (2.0 * mX_eV) + omega_eV[None, :] / q_eV[:, None]

        rel_diff = ((vmin_rel - vmin_halo) / vmin_halo).abs().max().item()
        assert rel_diff < 1e-6, f"v_min_relativistic deviates from halo formula by {rel_diff:.2e}"


class TestQBounds:
    def test_q_bounds_omega_zero(self):
        """When omega -> 0, q_min -> 0 and q_max -> 2 gamma mX v."""
        from DMeRates.srdm.kinematics import q_bounds, gamma

        mX_eV = 1e6
        v = torch.tensor([0.001, 0.01], dtype=torch.float64)
        omega = torch.tensor([1e-6], dtype=torch.float64)  # near-zero omega
        gam = gamma(v)

        q_min, q_max = q_bounds(v, omega, mX_eV)

        expected_q_max = 2.0 * gam * mX_eV * v
        for i in range(len(v)):
            assert float(q_min[i, 0]) == pytest.approx(0.0, abs=1.0)  # small but near zero
            assert float(q_max[i, 0]) == pytest.approx(float(expected_q_max[i]), rel=1e-4)

    def test_q_bounds_degenerate_no_nan(self):
        """When discriminant < 0, no NaN -- q_mask returns empty."""
        from DMeRates.srdm.kinematics import q_bounds, q_mask

        mX_eV = 1e4  # 10 keV
        v = torch.tensor([0.001], dtype=torch.float64)
        omega = torch.tensor([mX_eV * 2.0], dtype=torch.float64)  # above threshold

        q_min, q_max = q_bounds(v, omega, mX_eV)
        assert not torch.any(torch.isnan(q_min))
        assert not torch.any(torch.isnan(q_max))

        q_eV = torch.linspace(1.0, 1000.0, 50, dtype=torch.float64)
        mask = q_mask(q_eV, q_min, q_max)
        assert not torch.any(mask), "q_mask should be empty for degenerate kinematic case"


class TestHVector:
    def test_H_vector_nr_limit(self):
        """In NR limit (E_chi ~ mX, tiny omega), H_V ~ 4 mX^2."""
        from DMeRates.srdm.kinematics import H_vector

        mX_eV = 1e6
        q_eV = torch.tensor([10.0, 100.0], dtype=torch.float64)
        E_chi = torch.tensor(mX_eV, dtype=torch.float64)
        E_chi_prime = torch.tensor(mX_eV - 1.0, dtype=torch.float64)

        HV = H_vector(q_eV, E_chi, E_chi_prime)
        nr_limit = torch.tensor(4.0 * mX_eV**2, dtype=torch.float64) - q_eV**2
        rel_diff = ((HV - nr_limit) / nr_limit.abs()).abs().max().item()
        assert rel_diff < 1e-5, f"H_V deviates from NR limit by {rel_diff:.2e}"


class TestManifestFindEntry:
    def test_find_entry_exact_hit(self):
        """find_entry returns entry for exact key values."""
        from DMeRates.srdm.manifest import find_entry
        entry = find_entry(48232.9466, 1.098541e-38, 2, 'vector')
        assert entry is not None
        assert entry["grid_index"] == [10, 8]

    def test_find_entry_rtol_tight_hit(self):
        """find_entry hits when key perturbed by << rtol (1e-9 << 1e-6)."""
        from DMeRates.srdm.manifest import find_entry
        # 1e-9 relative perturbation -- well inside rtol=1e-6
        entry = find_entry(48232.9466 * (1 + 1e-9), 1.098541e-38, 2, 'vector')
        assert entry is not None, "Expected a hit for sub-rtol perturbation"

    def test_find_entry_rtol_large_miss(self):
        """find_entry misses when key differs by >> rtol (1e-3 >> 1e-6)."""
        from DMeRates.srdm.manifest import find_entry
        # 1e-3 relative perturbation -- far outside rtol=1e-6
        entry = find_entry(48232.9466 * (1 + 1e-3), 1.098541e-38, 2, 'vector')
        assert entry is None, "Expected None for large-rtol miss"

    def test_find_entry_nominal_values_miss(self):
        """Nominal round values (50 keV, 1e-38) do NOT match actual grid keys.

        This is intentional: the manifest uses actual grid values, not nominal
        ones. Downstream callers must read the manifest to discover the correct
        (mX_eV, sigma_e_cm2) keys. See entry['nominal_mX_eV'] for documentation.
        """
        from DMeRates.srdm.manifest import find_entry
        entry = find_entry(50000.0, 1e-38, 2, 'vector')
        assert entry is None, (
            "Nominal values should not match -- manifest uses actual grid values. "
            "Phase B callers must look up entry['mX_eV'] from the manifest."
        )


class TestFluxLoader:
    def test_load_srdm_flux_smoke(self):
        """load_srdm_flux with actual grid values returns sensible tensors.

        Uses mX=48232.9466 eV, sigma=1.098541e-38 cm^2 (DPLM grid row=10, col=8,
        closest to nominal 50 keV / 1e-38 cm^2).
        """
        import random
        random.seed(0)
        import numericalunits
        numericalunits.reset_units(seed=0)

        from DMeRates.srdm.flux_loader import load_srdm_flux
        # Use actual grid values recorded in manifest
        v_over_c, dphi_dv = load_srdm_flux(48232.9466, 1.098541e-38, 2, 'vector')

        assert v_over_c.shape == dphi_dv.shape
        assert v_over_c.ndim == 1
        assert len(v_over_c) > 10

        # SRDM velocities can reach ~0.165c for light DM (50 keV); well below 1.
        assert float(v_over_c.max()) < 1.0, f"Max v/c = {float(v_over_c.max()):.4f}, expected < 1.0 (physical)"
        assert float(v_over_c.max()) > 0.001, f"Max v/c suspiciously low: {float(v_over_c.max()):.4f}"
        assert len(v_over_c) == 299
        assert float(v_over_c.min()) > 0.0

        assert torch.all(torch.isfinite(dphi_dv))
        assert torch.all(dphi_dv >= 0)
        assert dphi_dv.max() > 0

    def test_load_srdm_flux_unit_conversion_is_bare_c_kms(self):
        """dPhi/d(v/c) conversion uses bare c[km/s], not randomized nu.c0."""
        import random
        random.seed(0)
        import numericalunits
        numericalunits.reset_units(seed=0)

        from DMeRates.srdm.flux_loader import load_srdm_flux

        raw = np.loadtxt("halo_data/srdm/srdm_dphidv_DPLM_row10_col8.txt", comments="#")
        v_over_c, dphi_dv = load_srdm_flux(48232.9466, 1.098541e-38, 2, 'vector')

        c_kms_bare = numericalunits.c0 / (numericalunits.km / numericalunits.s)
        expected_v0 = raw[1, 0] / c_kms_bare
        expected_dphi0 = raw[1, 1] * c_kms_bare
        got_dphi0 = float(dphi_dv[0] / (1.0 / (numericalunits.cm**2 * numericalunits.s)))

        assert float(v_over_c[0]) == pytest.approx(expected_v0, rel=1e-12)
        assert got_dphi0 == pytest.approx(expected_dphi0, rel=1e-12)

    def test_load_srdm_flux_smoke_second_entry(self):
        """load_srdm_flux with second manifest entry (0.5 MeV nominal, DPLM_15_20) works."""
        import random
        random.seed(0)
        import numericalunits
        numericalunits.reset_units(seed=0)

        from DMeRates.srdm.flux_loader import load_srdm_flux
        # Actual grid values for DPLM row=15, col=20
        v_over_c, dphi_dv = load_srdm_flux(510927.7440, 1.151395e-37, 2, 'vector')

        assert v_over_c.shape == dphi_dv.shape
        assert v_over_c.ndim == 1
        assert len(v_over_c) > 10
        assert float(v_over_c.max()) < 1.0
        assert float(v_over_c.max()) > 0.001
        assert float(v_over_c.min()) > 0.0
        assert torch.all(torch.isfinite(dphi_dv))
        assert dphi_dv.max() > 0

    def test_load_srdm_flux_non_vector_modes_reuse_vector_flux(self):
        """Non-vector detector modes reuse the vector flux file by policy."""
        from DMeRates.srdm.flux_loader import load_srdm_flux
        v_vec, phi_vec = load_srdm_flux(48232.9466, 1.098541e-38, 2, "vector")
        for spin in ("scalar", "approx", "approx_full", "approx full"):
            v_mode, phi_mode = load_srdm_flux(48232.9466, 1.098541e-38, 2, spin)
            assert torch.equal(v_mode, v_vec)
            assert torch.equal(phi_mode, phi_vec)

    def test_load_srdm_flux_miss_raises(self):
        """Unregistered tuple raises FileNotFoundError with tuple and manifest path."""
        import random
        random.seed(0)
        import numericalunits
        numericalunits.reset_units(seed=0)

        from DMeRates.srdm.flux_loader import load_srdm_flux
        from DMeRates.data.registry import DataRegistry

        with pytest.raises(FileNotFoundError) as exc_info:
            load_srdm_flux(1e3, 1e-40, 0, 'vector')

        msg = str(exc_info.value)
        assert str(DataRegistry.srdm_manifest()) in msg or 'manifest' in msg.lower()
