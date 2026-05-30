"""Round-trip tests for scripts/upstream_to_dmerates.py.

Each test builds a minimal synthetic upstream artifact in tmp_path, runs the
converter, then verifies that the relevant DMeRates loader layout is produced.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

_FLUX_ROWS = "100.0\t1.0\n200.0\t2.0\n300.0\t3.0\n"

_VERNE_META = {
    "schema": "verne_srdm_beam_flux_v1",
    "mDM_MeV": 10.0,
    "sigmaE_cm2": 1e-36,
    "sigmaP_cm2": 8.0e-36,
    "depth_m": 1400.0,
    "num_angles": 2,
    "angle_convention": {
        "internal_gamma_deg": "0 below, 180 overhead",
        "file_isoangle_deg": "0 overhead, 180 below",
    },
    "file_isoangle_deg": [0.18, 179.82],
    "gamma_internal_deg": [179.82, 0.18],
}

_DAMASCUS_META = {
    "flux_type": "post_earth_detector",
    "ring_count": 2,
    "angle_convention": "0=overhead, 90=horizon, 180=nadir",
    "angle_to_ring_mapping": "floor(angle_deg / 180.0 * ring_count), clamped to ring_count - 1",
    "m_chi_MeV": 1.0,
    "sigma_p_cm2": 8.72495270179e-35,
    "sigma_e_cm2": None,
    "detector_depth_m": 104.0,
    "site_label": "SENSEI",
}


def _make_verne_dir(tmp_path: Path) -> Path:
    d = tmp_path / "verne_run"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(_VERNE_META))
    for i in range(2):
        (d / f"Differential_SRDM_Flux_mDM_10.0000MeV_sigmaE_1e-36cm2_isoangle_{i:03d}.txt").write_text(
            _FLUX_ROWS
        )
    return d


def _make_damascus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "damascus_run"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(_DAMASCUS_META))
    for i in range(2):
        (d / f"Differential_SRDM_Flux_theta_{i}.txt").write_text(_FLUX_ROWS)
    return d


def _make_halo_eta_dir(tmp_path: Path, name: str = "halo_eta_run") -> Path:
    d = tmp_path / name
    d.mkdir()
    for i in range(2):
        (d / f"DM_Eta_theta_{i}.txt").write_text(
            f"100.0\t{1.0 + i}\t0.1\n200.0\t{2.0 + i}\t0.2\n"
        )
    return d


def _make_raw_damascus_halo_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw_damascus_halo"
    d.mkdir()
    (d / "sensei_test.rho").write_text("0\t0.6\n1\t0.3\n")
    hist = d / "sensei_test_histograms"
    hist.mkdir()
    # columns follow DaMaSCUS_helper.fix_eta: vmin, eta, unused, eta_err
    (hist / "eta.0").write_text("1.0\t2.0\t0.0\t0.2\n2.0\t4.0\t0.0\t0.4\n")
    (hist / "eta.1").write_text("1.0\t3.0\t0.0\t0.3\n2.0\t5.0\t0.0\t0.5\n")
    return d


# ---------------------------------------------------------------------------
# Converter output tests
# ---------------------------------------------------------------------------

def test_convert_verne_output_is_loadable(tmp_path):
    from scripts.upstream_to_dmerates import convert_verne
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux

    input_dir = _make_verne_dir(tmp_path)
    output_root = tmp_path / "halo_data"

    convert_verne(input_dir, output_root, FDMn=0)

    v, dphi, meta = load_srdmbeam_flux(
        10.0, 1e-36, 0, 0,
        modulated_source="Verne",
        base_data_dir=output_root,
    )

    assert torch.all(torch.isfinite(v))
    assert torch.all(torch.isfinite(dphi))
    assert torch.all(dphi >= 0)
    assert meta["modulated_source"] == "Verne"
    assert meta["ring_count"] == 2


def test_convert_verne_metadata_key_mapping(tmp_path):
    from scripts.upstream_to_dmerates import convert_verne
    from DMeRates.srdm.flux_loader import load_srdmbeam_metadata

    input_dir = _make_verne_dir(tmp_path)
    output_root = tmp_path / "halo_data"

    convert_verne(input_dir, output_root, FDMn=0)

    meta = load_srdmbeam_metadata(
        10.0, 1e-36, 0,
        modulated_source="Verne",
        base_data_dir=output_root,
    )

    assert meta["ring_count"] == 2
    assert meta["m_chi_MeV"] == pytest.approx(10.0)
    assert meta["sigma_e_cm2"] == pytest.approx(1e-36)
    assert meta["detector_depth_m"] == pytest.approx(1400.0)
    assert meta["angle_grid_type"] == "point"
    assert meta["file_isoangle_deg"] == pytest.approx([0.18, 179.82])
    assert isinstance(meta["raw_metadata"]["angle_convention"], str)


def test_convert_verne_flux_files_are_reachable_for_both_rings(tmp_path):
    from scripts.upstream_to_dmerates import convert_verne
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux

    input_dir = _make_verne_dir(tmp_path)
    output_root = tmp_path / "halo_data"
    convert_verne(input_dir, output_root, FDMn=0)

    for ring in range(2):
        v, dphi, _ = load_srdmbeam_flux(
            10.0, 1e-36, 0, ring,
            modulated_source="Verne",
            base_data_dir=output_root,
        )
        assert v.shape[0] == 3


def test_convert_damascus_output_is_loadable(tmp_path):
    from scripts.upstream_to_dmerates import convert_damascus
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux, format_srdmbeam_sigma
    from scripts.upstream_to_dmerates import _sigma_e_from_sigma_p

    sigma_e = _sigma_e_from_sigma_p(8.72495270179e-35, 1.0)
    input_dir = _make_damascus_dir(tmp_path)
    output_root = tmp_path / "halo_data"

    convert_damascus(input_dir, output_root, FDMn=2)

    v, dphi, meta = load_srdmbeam_flux(
        1.0, sigma_e, 2, 0,
        modulated_source="DaMaSCUS",
        base_data_dir=output_root,
    )

    assert torch.all(torch.isfinite(v))
    assert torch.all(torch.isfinite(dphi))
    assert meta["modulated_source"] == "DaMaSCUS"
    assert meta["ring_count"] == 2


def test_convert_damascus_sigma_e_derivation(tmp_path):
    from scripts.upstream_to_dmerates import convert_damascus, _sigma_e_from_sigma_p

    sigma_p = 8.72495270179e-35
    m_chi = 1.0
    expected_sigma_e = _sigma_e_from_sigma_p(sigma_p, m_chi)

    input_dir = _make_damascus_dir(tmp_path)
    output_root = tmp_path / "halo_data"
    out_dir = convert_damascus(input_dir, output_root, FDMn=2)

    written = json.loads((out_dir / "metadata.json").read_text())
    assert written["sigma_e_cm2"] == pytest.approx(expected_sigma_e, rel=1e-9)
    # Must be close to 1e-35 for these inputs (matches known Verne cross-section).
    assert written["sigma_e_cm2"] == pytest.approx(1e-35, rel=1e-3)


def test_convert_damascus_sigma_e_formula_reduced_masses():
    """Verify σ_e = σ_p·(μ_e/μ_p)² holds for a known mass."""
    from scripts.upstream_to_dmerates import _sigma_e_from_sigma_p, _M_E_MEV, _M_P_MEV

    m_chi = 1.0
    sigma_p = 8.72495270179e-35

    mu_e = m_chi * _M_E_MEV / (m_chi + _M_E_MEV)
    mu_p = m_chi * _M_P_MEV / (m_chi + _M_P_MEV)
    expected = sigma_p * (mu_e / mu_p) ** 2

    assert _sigma_e_from_sigma_p(sigma_p, m_chi) == pytest.approx(expected, rel=1e-15)


def test_convert_verne_missing_metadata_raises(tmp_path):
    from scripts.upstream_to_dmerates import convert_verne

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="metadata.json"):
        convert_verne(empty_dir, tmp_path / "out", FDMn=0)


def test_convert_damascus_missing_metadata_raises(tmp_path):
    from scripts.upstream_to_dmerates import convert_damascus

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="metadata.json"):
        convert_damascus(empty_dir, tmp_path / "out", FDMn=2)


def test_convert_verne_no_flux_files_raises(tmp_path):
    from scripts.upstream_to_dmerates import convert_verne

    d = tmp_path / "meta_only"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(_VERNE_META))

    with pytest.raises(FileNotFoundError, match="isoangle"):
        convert_verne(d, tmp_path / "out", FDMn=0)


def test_convert_damascus_no_flux_files_raises(tmp_path):
    from scripts.upstream_to_dmerates import convert_damascus

    d = tmp_path / "meta_only"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(_DAMASCUS_META))

    with pytest.raises(FileNotFoundError, match="theta"):
        convert_damascus(d, tmp_path / "out", FDMn=2)


def test_convert_halo_modulated_copies_verne_eta_layout(tmp_path):
    from scripts.upstream_to_dmerates import convert_halo_modulated

    input_dir = _make_halo_eta_dir(tmp_path)
    output_root = tmp_path / "halo_data"

    out_dir = convert_halo_modulated(
        input_dir,
        output_root,
        FDMn=2,
        source="verne",
        mX_MeV=1.0,
        sigma_e_cm2=1e-35,
    )

    assert out_dir == (
        output_root / "modulated" / "FDMq2" / "Verne"
        / "mDM_1_0_MeV_sigmaE_1e-35_cm2"
    )
    assert (out_dir / "DM_Eta_theta_0.txt").read_text().startswith("100.0\t1.0")
    assert (out_dir / "DM_Eta_theta_1.txt").read_text().startswith("100.0\t2.0")


def test_convert_halo_modulated_raw_damascus_applies_density_conversion(tmp_path):
    from scripts.upstream_to_dmerates import (
        _DAMASCUS_KM_NU,
        _DAMASCUS_S_NU,
        convert_halo_modulated,
    )

    input_dir = _make_raw_damascus_halo_dir(tmp_path)
    output_root = tmp_path / "halo_data"

    out_dir = convert_halo_modulated(
        input_dir,
        output_root,
        FDMn=0,
        source="damascus",
        mX_MeV=1.0,
        sigma_e_cm2=1e-36,
    )
    data0 = np.loadtxt(out_dir / "DM_Eta_theta_0.txt")
    data1 = np.loadtxt(out_dir / "DM_Eta_theta_1.txt")

    assert data0[0, 0] == pytest.approx(_DAMASCUS_S_NU / _DAMASCUS_KM_NU)
    assert data0[0, 1] == pytest.approx(
        2.0 * _DAMASCUS_KM_NU / _DAMASCUS_S_NU * (0.6 / 0.3)
    )
    assert data0[0, 2] == pytest.approx(
        0.2 * _DAMASCUS_KM_NU / _DAMASCUS_S_NU * (0.6 / 0.3)
    )
    assert data1[0, 1] == pytest.approx(
        3.0 * _DAMASCUS_KM_NU / _DAMASCUS_S_NU * (0.3 / 0.3)
    )
    written_meta = json.loads((out_dir / "metadata.json").read_text())
    assert written_meta["converted_from_raw_damascus"] is True


def test_convert_srdm_direct_registers_damascus_sun_manifest(tmp_path, monkeypatch):
    from DMeRates.data.registry import DataRegistry
    from DMeRates.srdm.flux_loader import load_srdm_flux
    from scripts.upstream_to_dmerates import convert_srdm_direct

    input_file = tmp_path / "Differential_SRDM_Flux.txt"
    input_file.write_text(_FLUX_ROWS)
    output_root = tmp_path / "halo_data"

    out_file = convert_srdm_direct(
        input_file,
        output_root,
        FDMn=2,
        source="damascus-sun",
        mX_MeV=1.0,
        sigma_e_cm2=1e-35,
        grid_family="fig22_test",
    )

    manifest = json.loads((output_root / "srdm" / "manifest.json").read_text())
    assert out_file.exists()
    assert manifest["files"][0]["source"] == "DaMaSCUS-SUN"
    assert manifest["files"][0]["grid_family"] == "fig22_test"
    monkeypatch.setattr(DataRegistry, "halo_root", output_root)
    v_over_c, dphi_dv = load_srdm_flux(1.0e6, 1e-35, 2, "vector")
    assert torch.all(torch.isfinite(v_over_c))
    assert torch.all(dphi_dv >= 0)


def test_convert_srdm_direct_replaces_same_physical_manifest_entry(tmp_path):
    from scripts.upstream_to_dmerates import convert_srdm_direct

    first = tmp_path / "first_flux.txt"
    second = tmp_path / "second_flux.txt"
    first.write_text(_FLUX_ROWS)
    second.write_text(_FLUX_ROWS)
    output_root = tmp_path / "halo_data"

    convert_srdm_direct(
        first,
        output_root,
        FDMn=2,
        source="damascus-sun",
        mX_MeV=1.0,
        sigma_e_cm2=1e-35,
        filename="first.txt",
    )
    convert_srdm_direct(
        second,
        output_root,
        FDMn=2,
        source="damascus-sun",
        mX_MeV=1.0,
        sigma_e_cm2=1e-35,
        filename="second.txt",
    )

    manifest = json.loads((output_root / "srdm" / "manifest.json").read_text())
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["filename"] == "second.txt"
