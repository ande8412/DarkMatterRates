"""Analytic Lindhard screening for legacy QEDark/QCDark1 form factors.

The implementation uses the zero-temperature 3D electron-gas/RPA dielectric
function from Lindhard's linear-response model and applies the semiconductor
rate correction |f_crys|^2 -> |f_crys|^2 / |epsilon_L(E, q)|^2.

For z=(E+i eta)/(q vF), u=q/(2 kF), and the retarded principal-log branch used
by torch, the dimensionless response is evaluated as

    G(z,u) = 1/2 - A(z-u) + A(z+u),
    A(x) = [1 - x^2] / (8u) log((x + 1) / (x - 1)).

This branch convention gives G(0,u)->1 for u<<1, so epsilon_L(q,0) approaches
1 + qTF^2/q^2 and the amplitude factor approaches q^2/(q^2+qTF^2).

References:
    J. Lindhard, Kgl. Danske Videnskab. Selskab Mat.-Fys. Medd. 28, No. 8
    (1954); arXiv:2404.10066 Sec. 2.5.3/App. D; arXiv:2306.14944 Eq. (11)
    and Table I for the Si/Ge analytical-screening constants also used here.
"""

from __future__ import annotations

import math

import numericalunits as nu
import torch

from DMeRates.Constants import me_eV, tf_screening


DEFAULT_LINDHARD_ETA_EV = 0.1


def lindhard_material_parameters(material: str) -> dict[str, float]:
    """Return bare-eV Lindhard parameters derived from QCDark1 Table-I inputs.

    The Table-I values provide omegaP and qTF for Si/Ge. We derive vF and kF
    from qTF^2 = 3 omegaP^2 / vF^2 so the static, small-q limit is exactly
    epsilon(q, 0) = 1 + qTF^2 / q^2.
    """
    if material not in tf_screening:
        raise ValueError(
            f"Lindhard screening is only configured for {sorted(tf_screening)}; "
            f"got material={material!r}."
        )

    params = tf_screening[material]
    omega_p_eV = float(params["omegaP"] / nu.eV)
    q_tf_eV = float(params["qTF"] / nu.eV)
    me_eV_bare = float(me_eV / nu.eV)
    v_f = math.sqrt(3.0) * omega_p_eV / q_tf_eV
    k_f_eV = me_eV_bare * v_f
    return {
        "omegaP_eV": omega_p_eV,
        "qTF_eV": q_tf_eV,
        "vF": v_f,
        "kF_eV": k_f_eV,
    }


def lindhard_dielectric(
    material: str,
    q: torch.Tensor,
    E: torch.Tensor,
    *,
    eta_eV: float = DEFAULT_LINDHARD_ETA_EV,
) -> torch.Tensor:
    """Return epsilon_L(q, E) on the tensor-product grid q x E.

    Args:
        material: Semiconductor material key, currently ``"Si"`` or ``"Ge"``.
        q: Momentum-transfer tensor with numericalunits momentum units.
        E: Energy tensor with numericalunits energy units.
        eta_eV: Positive imaginary broadening in eV for the Lindhard branch cuts.
    """
    params = lindhard_material_parameters(material)
    dtype = torch.get_default_dtype()
    q_eV = torch.as_tensor(q, dtype=dtype) / (nu.eV / nu.c0)
    E_eV = torch.as_tensor(E, dtype=dtype) / nu.eV

    q_eV = q_eV.flatten().unsqueeze(1)
    E_eV = E_eV.flatten().unsqueeze(0)
    q_eV = torch.clamp(q_eV, min=torch.finfo(dtype).tiny)

    q_tf = torch.as_tensor(params["qTF_eV"], dtype=dtype, device=q_eV.device)
    v_f = torch.as_tensor(params["vF"], dtype=dtype, device=q_eV.device)
    k_f = torch.as_tensor(params["kF_eV"], dtype=dtype, device=q_eV.device)
    eta = torch.as_tensor(float(eta_eV), dtype=dtype, device=q_eV.device)

    u_real = q_eV / (2.0 * k_f)
    z_real = E_eV / (q_eV * v_f)
    z_imag = eta / (q_eV * v_f)

    z = torch.complex(z_real, z_imag)
    u = torch.complex(u_real.expand_as(z_real), torch.zeros_like(z_real))
    one = torch.ones_like(z)

    term_minus = (one - (z - u) ** 2) / (8.0 * u)
    term_minus *= torch.log((z - u + one) / (z - u - one))
    term_plus = (one - (z + u) ** 2) / (8.0 * u)
    term_plus *= torch.log((z + u + one) / (z + u - one))
    # With the retarded (+i eta) prescription and principal complex logarithm,
    # this sign convention is the branch that yields G(0, u) -> 1 as u -> 0.
    G = 0.5 - term_minus + term_plus

    eps = one + torch.complex((q_tf / q_eV) ** 2, torch.zeros_like(q_eV)) * G
    return eps


def lindhard_screening(
    material: str,
    q: torch.Tensor,
    E: torch.Tensor,
    *,
    do_screen: bool = True,
    eta_eV: float = DEFAULT_LINDHARD_ETA_EV,
) -> torch.Tensor:
    """Return amplitude-level Lindhard screening, 1 / |epsilon_L(q, E)|."""
    if not do_screen:
        q_t = torch.as_tensor(q)
        E_t = torch.as_tensor(E)
        return torch.ones(
            (q_t.flatten().shape[0], E_t.flatten().shape[0]),
            dtype=torch.get_default_dtype(),
            device=q_t.device,
        )
    eps = lindhard_dielectric(material, q, E, eta_eV=eta_eV)
    return 1.0 / torch.abs(eps)


def lindhard_tfscreening(
    material: str,
    Earr: torch.Tensor,
    qArr: torch.Tensor,
    do_screen: bool,
    *,
    eta_eV: float = DEFAULT_LINDHARD_ETA_EV,
) -> torch.Tensor:
    """Return Lindhard screening on the legacy native E x q grid."""
    return lindhard_screening(
        material,
        qArr,
        Earr,
        do_screen=do_screen,
        eta_eV=eta_eV,
    ).T
