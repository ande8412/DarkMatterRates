from pathlib import Path

import json
import numpy as np
import numericalunits as nu
import pytest
import torch


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "srdmbeam"


def test_srdmbeam_load_returns_finite_tensors_and_metadata():
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux

    v_over_c, dphi_dv, metadata = load_srdmbeam_flux(
        100.0,
        1e-36,
        0,
        0,
        base_data_dir=FIXTURE_ROOT,
    )

    assert v_over_c.dtype == torch.float64
    assert dphi_dv.dtype == torch.float64
    assert v_over_c.shape == dphi_dv.shape
    assert v_over_c.ndim == 1
    assert torch.all(torch.isfinite(v_over_c))
    assert torch.all(torch.isfinite(dphi_dv))
    assert torch.all(dphi_dv >= 0)
    assert metadata["flux_source"] == "SRDMBeam"
    assert metadata["flux_type"] == "post_earth_detector"
    assert metadata["ring_index"] == 0
    assert metadata["ring_count"] == 4
    assert metadata["ring_isoangle_deg"] == pytest.approx(0.18)
    assert metadata["ring_representative_angle_deg"] == pytest.approx(0.18)


def test_srdmbeam_loader_drops_non_positive_velocity_rows():
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux

    v_over_c, _dphi_dv, _metadata = load_srdmbeam_flux(
        100.0,
        1e-36,
        0,
        0,
        base_data_dir=FIXTURE_ROOT,
    )

    c_kms_bare = nu.c0 / (nu.km / nu.s)
    assert len(v_over_c) == 3
    assert float(v_over_c[0]) == pytest.approx(100.0 / c_kms_bare, rel=1e-12)
    assert float(v_over_c.min()) > 0.0


def test_srdmbeam_total_flux_round_trips_to_raw_trapezoid():
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux, srdmbeam_flux_path

    v_over_c, dphi_dv, _metadata = load_srdmbeam_flux(
        100.0,
        1e-36,
        0,
        0,
        base_data_dir=FIXTURE_ROOT,
    )
    raw = np.loadtxt(
        srdmbeam_flux_path(100.0, 1e-36, 0, 0, base_data_dir=FIXTURE_ROOT),
        comments="#",
    )
    positive = raw[:, 0] > 0.0
    expected_total = np.trapezoid(raw[positive, 1], x=raw[positive, 0])

    recovered_weight = dphi_dv.detach().cpu().numpy() * float(nu.cm**2 * nu.s)
    recovered_total = np.trapezoid(recovered_weight, x=v_over_c.detach().cpu().numpy())

    assert recovered_total == pytest.approx(expected_total, rel=1e-12)


def test_srdmbeam_missing_file_error_includes_lookup_and_path():
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux, srdmbeam_flux_path

    attempted_path = srdmbeam_flux_path(
        100.0,
        1e-36,
        0,
        3,
        base_data_dir=FIXTURE_ROOT,
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        load_srdmbeam_flux(100.0, 1e-36, 0, 3, base_data_dir=FIXTURE_ROOT)

    msg = str(exc_info.value)
    assert "SRDMBeam" in msg
    assert "mX_MeV=100.0" in msg
    assert "sigma_e_cm2=1e-36" in msg
    assert "ring_index=3" in msg
    assert str(attempted_path) in msg


def test_srdmbeam_metadata_fields_are_loaded():
    from DMeRates.srdm.flux_loader import load_srdmbeam_metadata

    metadata = load_srdmbeam_metadata(
        100.0,
        1e-36,
        0,
        base_data_dir=FIXTURE_ROOT,
    )

    assert metadata["ring_count"] == 4
    assert metadata["angle_grid_type"] == "point"
    assert metadata["file_isoangle_deg"] == [0.18, 92.57, 128.0, 179.82]
    assert metadata["angle_representative_deg"] == [0.18, 92.57, 128.0, 179.82]
    assert metadata["gamma_internal_deg"] == [179.82, 87.43, 52.0, 0.18]
    assert metadata["angle_convention"] == "0=overhead, 90=horizon, 180=nadir"
    assert metadata["m_chi_MeV"] == 100.0
    assert metadata["sigma_e_cm2"] == 1e-36
    assert metadata["sigma_p_cm2"] == 2e-35
    assert metadata["site_label"] == "TEST"
    assert metadata["detector_depth_m"] == 1400.0
    assert metadata["raw_metadata"]["input_flux_file"] == "tests/fixtures/srdmbeam/input_fixture_flux.txt"


def test_srdmbeam_verne_source_uses_separated_directory(tmp_path):
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux, srdmbeam_parameter_dir

    parameter_dir = (
        tmp_path
        / "modulated"
        / "FDM1"
        / "Verne"
        / "SRDMBeam"
        / "mDM_10_0_MeV_sigmaE_1e-36_cm2"
    )
    parameter_dir.mkdir(parents=True)
    (parameter_dir / "metadata.json").write_text(
        json.dumps(
            {
                "flux_type": "post_earth_detector",
                "input_flux_file": "fixture.txt",
                "ring_count": 2,
                "angle_grid_type": "point",
                "file_isoangle_deg": [0.18, 179.82],
                "angle_convention": "0=overhead, 90=horizon, 180=nadir",
                "m_chi_MeV": 10.0,
                "sigma_p_cm2": 2.0e-35,
                "sigma_e_cm2": 1.0e-36,
                "site_label": "TEST",
                "detector_depth_m": 1400.0,
            }
        )
    )
    flux_file = (
        parameter_dir
        / "Differential_SRDM_Flux_mDM_10_0_MeV_sigmaE_1e-36_cm2_isoangle_1.txt"
    )
    flux_file.write_text("100 1.0\n200 2.0\n")

    resolved_dir = srdmbeam_parameter_dir(
        10.0,
        1e-36,
        0,
        modulated_source="Verne",
        base_data_dir=tmp_path,
    )
    _v_over_c, _dphi_dv, metadata = load_srdmbeam_flux(
        10.0,
        1e-36,
        0,
        1,
        modulated_source="Verne",
        base_data_dir=tmp_path,
    )

    assert resolved_dir == parameter_dir
    assert metadata["flux_source"] == "SRDMBeam"
    assert metadata["modulated_source"] == "Verne"
    assert metadata["upstream_source"] == "Verne"
    assert metadata["ring_representative_angle_deg"] == pytest.approx(179.82)
    assert "Verne/SRDMBeam" in metadata["parameter_dir"]


def test_srdmbeam_ring_count_uses_metadata_when_present():
    from DMeRates.srdm.flux_loader import load_srdmbeam_metadata

    metadata = load_srdmbeam_metadata(
        100.0,
        1e-36,
        0,
        base_data_dir=FIXTURE_ROOT,
    )

    assert metadata["ring_count"] == 4


def test_srdmbeam_metadata_infers_damascus_bin_centers(tmp_path):
    from DMeRates.srdm.flux_loader import load_srdmbeam_metadata

    parameter_dir = (
        tmp_path
        / "modulated"
        / "FDM1"
        / "SRDMBeam"
        / "mDM_10_0_MeV_sigmaE_1e-36_cm2"
    )
    parameter_dir.mkdir(parents=True)
    metadata_path = parameter_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "flux_type": "post_earth_detector",
                "input_flux_file": "fixture.txt",
                "ring_count": 6,
                "angle_convention": "0=overhead, 90=horizon, 180=nadir",
                "angle_to_ring_mapping": (
                    "floor(angle_deg / 180.0 * ring_count), "
                    "clamped to ring_count - 1"
                ),
                "m_chi_MeV": 10.0,
                "sigma_p_cm2": 2.0e-35,
                "sigma_e_cm2": 1.0e-36,
                "site_label": "TEST",
                "detector_depth_m": 1400.0,
            }
        )
    )

    with pytest.warns(RuntimeWarning, match="inferring bin-average"):
        metadata = load_srdmbeam_metadata(
            10.0,
            1e-36,
            0,
            base_data_dir=tmp_path,
        )

    assert metadata["angle_grid_type"] == "bin_average"
    assert metadata["file_isoangle_deg"] is None
    assert metadata["angle_bin_edges_deg"] == [
        0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0
    ]
    assert metadata["angle_bin_centers_deg"] == [
        15.0, 45.0, 75.0, 105.0, 135.0, 165.0
    ]
    assert metadata["angle_representative_deg"] == [
        15.0, 45.0, 75.0, 105.0, 135.0, 165.0
    ]


def test_srdmbeam_ring_count_falls_back_to_fixture_file_count():
    from DMeRates.srdm.flux_loader import available_srdmbeam_ring_indices, load_srdmbeam_metadata

    with pytest.warns(RuntimeWarning, match="metadata.json is missing"):
        metadata = load_srdmbeam_metadata(
            50.0,
            3.5e-38,
            2,
            base_data_dir=FIXTURE_ROOT,
        )

    assert metadata["ring_count"] == 2
    assert available_srdmbeam_ring_indices(
        50.0,
        3.5e-38,
        2,
        base_data_dir=FIXTURE_ROOT,
    ) == [0, 1]


def test_srdmbeam_rejects_unsupported_fdmn():
    from DMeRates.srdm.flux_loader import srdmbeam_parameter_dir

    with pytest.raises(ValueError, match="FDMn=1"):
        srdmbeam_parameter_dir(100.0, 1e-36, 1, base_data_dir=FIXTURE_ROOT)


def test_direct_load_srdm_flux_still_returns_two_values():
    from DMeRates.srdm.flux_loader import load_srdm_flux

    loaded = load_srdm_flux(48232.9466, 1.098541e-38, 2, "vector")

    assert len(loaded) == 2
    assert loaded[0].shape == loaded[1].shape


def test_damascus_bin_average_metadata_fields():
    from DMeRates.srdm.flux_loader import load_srdmbeam_metadata

    metadata = load_srdmbeam_metadata(
        100.0,
        1e-36,
        0,
        modulated_source="DaMaSCUS",
        base_data_dir=FIXTURE_ROOT,
    )

    assert metadata["angle_grid_type"] == "bin_average"
    assert metadata["angle_bin_edges_deg"] == [0.0, 45.0, 90.0, 135.0, 180.0]
    assert metadata["angle_bin_centers_deg"] == [22.5, 67.5, 112.5, 157.5]
    assert metadata["angle_representative_deg"] == metadata["angle_bin_centers_deg"]
    assert metadata["file_isoangle_deg"] is None
    assert metadata["ring_count"] == 4
    assert metadata["modulated_source"] == "DaMaSCUS"


def test_damascus_bin_average_ring_representative_angle():
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux

    v_over_c, dphi_dv, metadata = load_srdmbeam_flux(
        100.0,
        1e-36,
        0,
        0,
        modulated_source="DaMaSCUS",
        base_data_dir=FIXTURE_ROOT,
    )

    assert torch.all(torch.isfinite(v_over_c))
    assert torch.all(torch.isfinite(dphi_dv))
    assert metadata["ring_representative_angle_deg"] == pytest.approx(22.5)
    assert "ring_isoangle_deg" not in metadata or metadata.get("ring_isoangle_deg") is None


def test_damascus_bin_average_source_resolves_to_correct_directory():
    from DMeRates.srdm.flux_loader import srdmbeam_parameter_dir

    expected = (
        FIXTURE_ROOT
        / "modulated"
        / "FDM1"
        / "DaMaSCUS"
        / "SRDMBeam"
        / "mDM_100_0_MeV_sigmaE_1e-36_cm2"
    )
    resolved = srdmbeam_parameter_dir(
        100.0,
        1e-36,
        0,
        modulated_source="DaMaSCUS",
        base_data_dir=FIXTURE_ROOT,
    )

    assert resolved == expected
    assert "DaMaSCUS/SRDMBeam" in str(resolved)


def test_srdmbeam_verne_separated_fixture_loads():
    from DMeRates.srdm.flux_loader import load_srdmbeam_flux

    verne_fixture_root = FIXTURE_ROOT / "modulated" / "FDM1" / "Verne" / "SRDMBeam"
    if not (verne_fixture_root / "mDM_100_0_MeV_sigmaE_1e-36_cm2").exists():
        pytest.skip("Verne separated fixture not found")

    v_over_c, dphi_dv, metadata = load_srdmbeam_flux(
        100.0,
        1e-36,
        0,
        0,
        modulated_source="Verne",
        base_data_dir=FIXTURE_ROOT,
    )

    assert torch.all(torch.isfinite(v_over_c))
    assert torch.all(torch.isfinite(dphi_dv))
    assert metadata["modulated_source"] == "Verne"
    assert metadata["angle_grid_type"] == "point"
    assert metadata["angle_representative_deg"] == [0.18, 45.0, 90.0, 135.0]
