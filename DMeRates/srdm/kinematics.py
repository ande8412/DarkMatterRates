"""Relativistic SRDM kinematics for DMeRates engines.

All quantities in eV (energy/mass) and dimensionless v/c.
No numericalunits -- this module lives inside the vectorized hot path.

References:
    QCDark2 paper: arXiv:2603.12326 (Hochberg et al. 2026)
        eq. 2.6  -- spin-structure factors H_V, H_phi
        eq. 2.9  -- relativistic v_min
        eq. A.19 -- q_min / q_max bounds
        eq. A.20 -- mediator propagator prefactor
"""
import torch
import math
from DMeRates.srdm.mediators import normalize_mediator_spin


def gamma(v_over_c: torch.Tensor) -> torch.Tensor:
    """Lorentz factor gamma = 1/sqrt(1 - v^2).

    Args:
        v_over_c: dimensionless velocity tensor, any shape.

    Returns:
        gamma tensor, same shape.
    """
    return 1.0 / torch.sqrt(1.0 - v_over_c**2)


def v_min_relativistic(
    q_eV: torch.Tensor,
    omega_eV: torch.Tensor,
    mX_eV: float,
    gamma_v: torch.Tensor,
) -> torch.Tensor:
    """Relativistic v_min for DM-electron scattering (QCDark2 eq. 2.9).

    v_min(q, omega) = q / (2 gamma m_chi) + omega / q

    Args:
        q_eV     : DM momentum transfer, shape (N_q,) or broadcastable.
        omega_eV : energy transfer, shape (N_E,) or broadcastable.
        mX_eV    : DM mass in eV (scalar).
        gamma_v  : Lorentz factor, shape (N_v,) or broadcastable.

    Returns:
        v_min tensor broadcast over all input shapes.
    """
    return q_eV / (2.0 * gamma_v * mX_eV) + omega_eV / q_eV


def q_bounds(
    v_over_c: torch.Tensor,
    omega_eV: torch.Tensor,
    mX_eV: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q_min and q_max for given velocity and energy transfer (QCDark2 eq. A.19).

    q_min = gamma m_chi v - sqrt((gamma m_chi - omega)^2 - m_chi^2)
    q_max = gamma m_chi v + sqrt((gamma m_chi - omega)^2 - m_chi^2)

    Where (gamma m_chi - omega)^2 < m_chi^2, sets q_max < q_min so the
    q_mask zeros the contribution without NaN.

    Args:
        v_over_c : shape (N_v,)
        omega_eV : shape (N_E,)
        mX_eV    : DM mass in eV (scalar)

    Returns:
        q_min : shape (N_v, N_E)
        q_max : shape (N_v, N_E)
    """
    v = v_over_c[:, None]        # (N_v, 1)
    omega = omega_eV[None, :]    # (1, N_E)

    gam = 1.0 / torch.sqrt(1.0 - v**2)
    gam_mX = gam * mX_eV
    centre = gam_mX - omega      # (N_v, N_E)

    discriminant = centre**2 - mX_eV**2
    # Where discriminant < 0, sqrt would be NaN -- set it to inf so q_max < q_min
    sqrt_disc = torch.where(
        discriminant >= 0,
        torch.sqrt(torch.clamp(discriminant, min=0.0)),
        torch.full_like(discriminant, float('inf')),
    )

    mid = gam_mX * v             # (N_v, N_E)
    q_min = mid - sqrt_disc
    q_max = mid + sqrt_disc

    # For discriminant < 0: q_max < 0 < q_min -> empty interval
    q_max = torch.where(discriminant >= 0, q_max, torch.full_like(q_max, -1.0))
    q_min = torch.where(discriminant >= 0, q_min, torch.zeros_like(q_min))

    return q_min, q_max


def q_mask(
    q_eV: torch.Tensor,
    q_min: torch.Tensor,
    q_max: torch.Tensor,
) -> torch.Tensor:
    """Bool mask zeroing integrand outside [q_min, q_max].

    Args:
        q_eV  : shape (N_q,)
        q_min : shape (N_v, N_E)
        q_max : shape (N_v, N_E)

    Returns:
        mask : bool tensor shape (N_v, N_q, N_E)
    """
    q = q_eV[None, :, None]           # (1, N_q, 1)
    qlo = q_min[:, None, :]           # (N_v, 1, N_E)
    qhi = q_max[:, None, :]           # (N_v, 1, N_E)
    return (q >= qlo) & (q <= qhi)


def H_vector(
    q_eV: torch.Tensor,
    E_chi_eV: torch.Tensor,
    E_chi_prime_eV: torch.Tensor,
) -> torch.Tensor:
    """Spin-structure factor for vector mediator H_V (QCDark2 eq. 2.6).

    H_V = (E_chi + E_chi')^2 - q^2

    Args:
        q_eV           : DM momentum, any broadcastable shape.
        E_chi_eV       : initial DM energy E_chi = gamma * m_chi.
        E_chi_prime_eV : final DM energy E_chi' = E_chi - omega.

    Returns:
        H_V tensor.
    """
    return (E_chi_eV + E_chi_prime_eV)**2 - q_eV**2


def H_scalar(
    q_eV: torch.Tensor,
    E_chi_eV: torch.Tensor,
    E_chi_prime_eV: torch.Tensor,
    mX_eV: float,
) -> torch.Tensor:
    """Spin-structure factor for scalar mediator H_phi (QCDark2 eq. 2.6).

    H_phi = 4 m_chi^2 - (E_chi - E_chi')^2 + q^2

    Args:
        q_eV           : DM momentum, any broadcastable shape.
        E_chi_eV       : initial DM energy.
        E_chi_prime_eV : final DM energy.
        mX_eV          : DM mass in eV.

    Returns:
        H_phi tensor.
    """
    return 4.0 * mX_eV**2 - (E_chi_eV - E_chi_prime_eV)**2 + q_eV**2


def mediator_propagator_inv_sq(
    q_eV: torch.Tensor,
    omega_eV: torch.Tensor,
    mA_eV: float,
) -> torch.Tensor:
    """Inverse squared mediator propagator 1/(omega^2 - q^2 - m_A^2)^2 (QCDark2 eq. A.20 denominator).

    For the validated semiconductor SRDM grids, q is O(10^1-10^5 eV) while
    omega is O(10^0-10^2 eV), and the q-bounds mask removes inaccessible
    points before integration. A future grid that can hit an exact propagator
    pole should mask or regulate that point at the engine boundary.

    Args:
        q_eV    : DM momentum, shape (N_q,) or broadcastable.
        omega_eV: energy transfer, shape (N_E,) or broadcastable.
        mA_eV   : mediator mass in eV (0.0 for massless / light mediator).

    Returns:
        1 / (omega^2 - q^2 - m_A^2)^2 tensor.
    """
    denom = omega_eV**2 - q_eV**2 - mA_eV**2
    return 1.0 / denom**2


def reference_propagator_factor(mA_eV: float, alpha_FS: float, me_eV: float) -> float:
    """Prefactor (m_A^2 + (alpha m_e)^2)^2 from sigma_bar_e definition (QCDark2 eq. A.20).

    This factor appears in the sigma_bar_e reference cross section to make the
    rate independent of mediator mass in the heavy/light mediator limits.

    Args:
        mA_eV   : mediator mass in eV.
        alpha_FS: fine structure constant (dimensionless).
        me_eV   : electron mass in eV.

    Returns:
        (m_A^2 + (alpha * m_e)^2)^2 as a float.
    """
    alpha_me_sq = (alpha_FS * me_eV) ** 2
    return (mA_eV**2 + alpha_me_sq) ** 2


def srdm_integrand_kernel(
    q_eV: torch.Tensor,
    omega_eV: torch.Tensor,
    E_chi_eV: torch.Tensor,
    mX_eV: float,
    mA_eV: float,
    mediator_spin: str,
) -> torch.Tensor:
    """Return the SRDM detector kernel matching QCDark2 mediator modes.

    Args:
        q_eV: momentum-transfer tensor, broadcastable to (N_v, N_q, N_E).
        omega_eV: energy-transfer tensor, broadcastable to (N_v, N_q, N_E).
        E_chi_eV: incoming DM energy tensor, broadcastable to (N_v, N_q, N_E).
        mX_eV: DM mass in eV.
        mA_eV: mediator mass in eV.
        mediator_spin: public mediator mode name (or alias `approx full`).
    """
    mode = normalize_mediator_spin(mediator_spin)
    if mode == "approx_full":
        gamma = E_chi_eV / mX_eV
        return (
            q_eV**3
            / mX_eV
            * (4.0 * mX_eV**2)
            / (q_eV**2 + mA_eV**2) ** 2
            * gamma
        )

    propagator = mediator_propagator_inv_sq(q_eV, omega_eV, mA_eV)
    common = q_eV**3 / (E_chi_eV - omega_eV) * propagator

    if mode == "vector":
        numerator = (2.0 * E_chi_eV - omega_eV) ** 2 - q_eV**2
    elif mode == "scalar":
        numerator = q_eV**2 - omega_eV**2 + 4.0 * mX_eV**2
    elif mode == "approx":
        numerator = torch.full_like(q_eV, 4.0 * mX_eV**2)
    else:
        raise ValueError(f"Unsupported mediator_spin={mediator_spin!r}.")

    return common * numerator
