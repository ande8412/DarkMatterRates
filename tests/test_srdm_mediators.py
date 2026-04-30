import pytest

from DMeRates.srdm.mediators import (
    flux_mediator_spin,
    normalize_mediator_spin,
    supported_mediator_spins,
)


def test_supported_mediator_spins_public_contract():
    assert supported_mediator_spins() == ("vector", "scalar", "approx", "approx_full")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("vector", "vector"),
        ("scalar", "scalar"),
        ("approx", "approx"),
        ("approx_full", "approx_full"),
        ("approx full", "approx_full"),
        ("  APPROX FULL  ", "approx_full"),
    ],
)
def test_normalize_mediator_spin(raw, expected):
    assert normalize_mediator_spin(raw) == expected


@pytest.mark.parametrize("raw", ["", "foo", "approxfull", "vectorial"])
def test_normalize_mediator_spin_invalid(raw):
    with pytest.raises(ValueError, match="Supported: vector, scalar, approx, approx_full"):
        normalize_mediator_spin(raw)


@pytest.mark.parametrize("mode", ["vector", "scalar", "approx", "approx_full", "approx full"])
def test_flux_mediator_spin_policy(mode):
    assert flux_mediator_spin(mode) == "vector"

