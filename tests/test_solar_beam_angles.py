"""Tests for Step D2 solar-beam SRDM angle utilities.

These helpers map solar altitude to the file-facing srdm_isoangle_deg used
for SRDMBeam flux lookup.  They are distinct from ThetaIso() which computes
the galactic halo-wind isodetection angle.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULATION_DIR = REPO_ROOT / "modulation_study"
if str(MODULATION_DIR) not in sys.path:
    sys.path.insert(0, str(MODULATION_DIR))

import isoangle  # noqa: E402 — needed so monkeypatch works on module attributes
from isoangle import (  # noqa: E402
    SolarBeamIsoAngle,
    SolarBeamInternalGamma,
    angle_to_ring_index,
    get_solar_angle_limits,
    solar_daily_angle_series,
    get_site_location,
    normalize_site_key,
    _srdm_iso_from_alt_deg,
    _internal_gamma_from_alt_deg,
)


# ---------------------------------------------------------------------------
# Pure angle-formula helpers — no Astropy required
# ---------------------------------------------------------------------------

class TestAngleFormulas:
    def test_overhead_iso(self):
        assert _srdm_iso_from_alt_deg(90.0) == pytest.approx(0.0)

    def test_horizon_iso(self):
        assert _srdm_iso_from_alt_deg(0.0) == pytest.approx(90.0)

    def test_nadir_iso(self):
        assert _srdm_iso_from_alt_deg(-90.0) == pytest.approx(180.0)

    def test_overhead_gamma(self):
        assert _internal_gamma_from_alt_deg(90.0) == pytest.approx(180.0)

    def test_horizon_gamma(self):
        assert _internal_gamma_from_alt_deg(0.0) == pytest.approx(90.0)

    def test_nadir_gamma(self):
        assert _internal_gamma_from_alt_deg(-90.0) == pytest.approx(0.0)

    def test_iso_and_gamma_are_complementary(self):
        for alt in [-90, -45, 0, 45, 90]:
            iso = _srdm_iso_from_alt_deg(alt)
            gamma = _internal_gamma_from_alt_deg(alt)
            assert iso + gamma == pytest.approx(180.0)

    def test_clip_above_90(self):
        # altitude above physical maximum — result must be clipped to 0
        assert _srdm_iso_from_alt_deg(95.0) == pytest.approx(0.0)

    def test_clip_below_minus_90(self):
        # altitude below physical minimum — result must be clipped to 180
        assert _srdm_iso_from_alt_deg(-95.0) == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# angle_to_ring_index exact boundaries
# ---------------------------------------------------------------------------

class TestAngleToRingIndex:
    def test_zero_maps_to_ring_zero(self):
        assert angle_to_ring_index(0.0, 36) == 0

    def test_just_below_180_maps_to_last_ring(self):
        # 179.999...° should still be ring 35, not 36
        assert angle_to_ring_index(180.0 - 1e-9, 36) == 35

    def test_exactly_180_maps_to_last_ring(self):
        assert angle_to_ring_index(180.0, 36) == 35

    def test_negative_clipped_to_zero(self):
        assert angle_to_ring_index(-5.0, 36) == 0

    def test_above_180_clipped_to_last_ring(self):
        assert angle_to_ring_index(200.0, 36) == 35

    def test_midpoint_90_deg_ring_count_36(self):
        # 90 / 180 * 36 = 18
        assert angle_to_ring_index(90.0, 36) == 18

    def test_ring_count_1_always_returns_zero(self):
        for angle in [0.0, 90.0, 180.0]:
            assert angle_to_ring_index(angle, 1) == 0

    def test_ring_count_zero_raises(self):
        with pytest.raises(ValueError):
            angle_to_ring_index(90.0, 0)

    def test_ring_count_negative_raises(self):
        with pytest.raises(ValueError):
            angle_to_ring_index(90.0, -1)

    def test_boundary_at_each_ring_edge(self):
        ring_count = 10
        for i in range(ring_count):
            lower = i * 180.0 / ring_count
            assert angle_to_ring_index(lower, ring_count) == i

    def test_ring_count_not_hardcoded(self):
        # Verify the formula works for ring_count values other than 36
        assert angle_to_ring_index(60.0, 9) == 3   # 60/180*9=3
        assert angle_to_ring_index(120.0, 6) == 4  # 120/180*6=4


# ---------------------------------------------------------------------------
# Public helpers with mocked solar altitude — tests the angle chain
# ---------------------------------------------------------------------------

class TestPublicHelpersWithMockedAltitude:
    def _patch_alt(self, monkeypatch, alt_deg):
        monkeypatch.setattr(isoangle, "_solar_alt_deg", lambda loc, t: alt_deg)

    def test_solar_beam_iso_overhead(self, monkeypatch):
        self._patch_alt(monkeypatch, 90.0)
        t = Time("2024-06-21T12:00:00", format="isot", scale="utc")
        assert SolarBeamIsoAngle("SNO", t) == pytest.approx(0.0)

    def test_solar_beam_iso_horizon(self, monkeypatch):
        self._patch_alt(monkeypatch, 0.0)
        t = Time("2024-06-21T06:00:00", format="isot", scale="utc")
        assert SolarBeamIsoAngle("SNO", t) == pytest.approx(90.0)

    def test_solar_beam_iso_nadir(self, monkeypatch):
        self._patch_alt(monkeypatch, -90.0)
        t = Time("2024-06-21T00:00:00", format="isot", scale="utc")
        assert SolarBeamIsoAngle("SNO", t) == pytest.approx(180.0)

    def test_solar_beam_internal_gamma_overhead(self, monkeypatch):
        self._patch_alt(monkeypatch, 90.0)
        t = Time("2024-06-21T12:00:00", format="isot", scale="utc")
        assert SolarBeamInternalGamma("SNO", t) == pytest.approx(180.0)

    def test_solar_beam_internal_gamma_nadir(self, monkeypatch):
        self._patch_alt(monkeypatch, -90.0)
        t = Time("2024-06-21T00:00:00", format="isot", scale="utc")
        assert SolarBeamInternalGamma("SNO", t) == pytest.approx(0.0)

    def test_internal_gamma_not_used_for_ring_lookup(self, monkeypatch):
        # SolarBeamInternalGamma returns gamma, not iso; they differ except at horizon
        self._patch_alt(monkeypatch, 45.0)
        t = Time("2024-06-21T09:00:00", format="isot", scale="utc")
        iso = SolarBeamIsoAngle("SNO", t)
        gamma = SolarBeamInternalGamma("SNO", t)
        assert iso == pytest.approx(45.0)
        assert gamma == pytest.approx(135.0)
        assert iso != gamma


# ---------------------------------------------------------------------------
# Site alias resolution
# ---------------------------------------------------------------------------

class TestSiteAliasResolution:
    def test_snolab_alias_normalizes_to_sno(self):
        assert normalize_site_key("SNOLAB") == "SNO"

    def test_snolab_location_matches_sno_location(self):
        snolab = get_site_location("SNOLAB")
        sno = get_site_location("SNO")
        assert snolab.lat.deg == pytest.approx(sno.lat.deg)
        assert snolab.lon.deg == pytest.approx(sno.lon.deg)

    def test_solar_beam_angle_snolab_matches_sno(self, monkeypatch):
        monkeypatch.setattr(isoangle, "_solar_alt_deg", lambda loc, t: 30.0)
        t = Time("2024-06-21T12:00:00", format="isot", scale="utc")
        assert SolarBeamIsoAngle("SNOLAB", t) == pytest.approx(
            SolarBeamIsoAngle("SNO", t)
        )


# ---------------------------------------------------------------------------
# No network lookup
# ---------------------------------------------------------------------------

class TestNoNetworkLookup:
    def test_get_site_location_no_network(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("EarthLocation.of_address must not be called")

        monkeypatch.setattr(EarthLocation, "of_address", fail)
        loc = get_site_location("SNOLAB")
        assert isinstance(loc, EarthLocation)

    def test_solar_beam_angle_no_network(self, monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("EarthLocation.of_address must not be called")

        monkeypatch.setattr(EarthLocation, "of_address", fail)
        monkeypatch.setattr(isoangle, "_solar_alt_deg", lambda loc, t: 0.0)
        t = Time("2024-06-21T06:00:00", format="isot", scale="utc")
        result = SolarBeamIsoAngle("SNOLAB", t)
        assert result == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# solar_daily_angle_series and get_solar_angle_limits
# ---------------------------------------------------------------------------

class TestSolarDailySeries:
    def test_returns_correct_types(self):
        times, angles = solar_daily_angle_series("SNO", [21, 6, 2024], cadence_minutes=60)
        assert isinstance(times, Time)
        assert isinstance(angles, np.ndarray)

    def test_cadence_60_gives_24_samples(self):
        times, angles = solar_daily_angle_series("SNO", [21, 6, 2024], cadence_minutes=60)
        assert len(times) == 24
        assert len(angles) == 24

    def test_angles_within_valid_range(self):
        _, angles = solar_daily_angle_series("SNO", [21, 6, 2024], cadence_minutes=30)
        assert np.all(angles >= 0.0)
        assert np.all(angles <= 180.0)

    def test_daily_limits_within_range(self):
        mn, mx = get_solar_angle_limits("SNO", [21, 6, 2024])
        assert 0.0 <= mn <= mx <= 180.0

    def test_daily_limits_snolab_alias(self):
        limits_snolab = get_solar_angle_limits("SNOLAB", [21, 6, 2024])
        limits_sno = get_solar_angle_limits("SNO", [21, 6, 2024])
        assert limits_snolab == pytest.approx(limits_sno)

    def test_accepts_earthlocation(self):
        loc = get_site_location("SNO")
        mn, mx = get_solar_angle_limits(loc, [21, 6, 2024])
        assert 0.0 <= mn <= mx <= 180.0

    def test_times_span_one_day(self):
        times, _ = solar_daily_angle_series("SNO", [21, 6, 2024], cadence_minutes=60)
        span_hours = (times[-1] - times[0]).to_value("hr")
        assert span_hours == pytest.approx(23.0, abs=0.1)  # 23 gaps of 1 hour

    def test_date_format_day_month_year(self):
        # Verify [day, month, year] order matches existing get_angle_limits convention
        _, angles_june = solar_daily_angle_series("SNO", [21, 6, 2024], cadence_minutes=60)
        _, angles_dec = solar_daily_angle_series("SNO", [21, 12, 2024], cadence_minutes=60)
        # SNO (SNOLAB, northern hemisphere) — summer solstice sun reaches higher
        # so minimum srdm_isoangle_deg should be smaller in June than in December
        assert np.min(angles_june) < np.min(angles_dec)
