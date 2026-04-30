"""Native QCDark2 dielectric engine for DMeRates.

Computes dR/dE for dark-matter electron scattering in semiconductors using
QCDark2 HDF5 dielectric data plus DMeRates halo providers. Production code
deliberately does NOT import qcdark2.* — the formula is implemented from the
derivation in `tests/qcdark2_formula_derivation.ipynb` (Plan Step 3.0), which
reproduces QCDark2's `get_dR_dE()` to machine precision.

Convention (mirrors QCDark2 internal natural units):
- q in alpha*m_e momentum units (a.k.a. q_ame); q_eV = q_ame * alpha * (m_e c^2)
- E, dE in eV
- masses in eV (rest energy)
- rho_X as ENERGY density (eV/cm^3)
- sigma_e in cm^2
- v expressed as v/c (dimensionless)

The single boundary where we lift to numericalunits is the call into
`DM_Halo_Distributions.eta_MB_tensor`, which expects nu-velocity in and returns
1/(nu-velocity); we convert in/out with `nu.c0` to match the bare-float
convention used everywhere else.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import numericalunits as nu
import scipy.integrate as _spi
import torch

from DMeRates.responses.dielectric import dielectric_response
from DMeRates.screening.dielectric import (
    dielectric_screening_ratio,
    normalize_dielectric_screening,
)
from DMeRates.spectrum import RateSpectrum
from DMeRates.srdm.mediators import (
    flux_mediator_spin as _flux_mediator_spin,
    normalize_mediator_spin as _normalize_mediator_spin,
)

_VALID_MEDIATORS = {0, 2}  # FDMn
_LARGE_MA_EV = 1e10  # proxy for m_A → ∞ in heavy-mediator limit; >> max q in any grid


# ----------------------------------------------------------------------------
# Step 1 utilities — pure functions, all bare-float (no nu inside).
# ----------------------------------------------------------------------------

def energy_loss_function(epsilon):
    """Energy-loss function ELF(q,E) = Im(eps) / (Im(eps)^2 + Re(eps)^2).

    Args:
        epsilon: complex ndarray, shape (N_q, N_E)
    Returns:
        real ndarray, shape (N_q, N_E)
    """
    im = np.imag(epsilon)
    re = np.real(epsilon)
    return im / (im * im + re * re)


def dynamic_structure_factor(epsilon, q_ame, alpha=None):
    """QCDark2 dynamic structure factor S(q,E) = ELF * q_ame^2 / (2 pi alpha).

    q is in alpha*m_e units (a.k.a. q_ame); the prefactor is dimensionless.

    Args:
        epsilon: complex ndarray, shape (N_q, N_E)
        q_ame: real ndarray, shape (N_q,) in alpha*m_e momentum units
        alpha: fine-structure constant; defaults to nu.alphaFS (a pure number).
    Returns:
        real ndarray, shape (N_q, N_E)
    """
    if alpha is None:
        alpha = nu.alphaFS
    elf = energy_loss_function(epsilon)
    return elf * (q_ame[:, None] ** 2) / (2.0 * np.pi * alpha)


def mediator_factor(q_ame, FDMn, alpha=None):
    """F_DM(q)^2 for a generic power-law mediator.

    FDMn=0 (heavy):  F_DM = 1                 -> ones(N_q)
    FDMn=2 (light):  F_DM = (alpha m_e / q)^2 -> (1/q_ame^2)^2 = q_ame^-4

    Note that F_DM has units alpha*m_e / q, so when q is already in alpha*m_e
    units the ratio reduces to 1/q_ame**FDMn for the form factor and
    1/q_ame**(2*FDMn) for F_DM^2.

    Args:
        q_ame: real ndarray, shape (N_q,) in alpha*m_e units
        FDMn: 0 (heavy mediator) or 2 (light mediator)
        alpha: unused for FDMn in {0,2}; accepted for API symmetry.
    Returns:
        real ndarray, shape (N_q,)
    """
    if FDMn not in _VALID_MEDIATORS:
        raise ValueError(
            f"Unsupported FDMn={FDMn}. Supported: {sorted(_VALID_MEDIATORS)}"
        )
    if FDMn == 0:
        return np.ones_like(q_ame)
    # F_DM = (alpha * m_e / q)^n. With q expressed in alpha*m_e units, F_DM = q_ame^-n.
    return q_ame ** (-2 * FDMn)


def v_min_bare(q_eV, E_eV, mX_eV):
    """v_min(q, E) / c in QCDark2 bare-float convention.

    Returns shape (N_q, N_E), value = q_eV / (2 mX_eV) + E_eV / q_eV.
    All inputs are in eV; the result is dimensionless v/c.
    """
    return q_eV[:, None] / (2.0 * mX_eV) + E_eV[None, :] / q_eV[:, None]


# ----------------------------------------------------------------------------
# Halo dispatch — uses DMeRates DM_Halo_Distributions for 'imb' (Maxwell-Boltzmann).
# ----------------------------------------------------------------------------

def _eta_imb_bare(vmin_over_c, halo_distribution):
    """Evaluate the DMeRates 'imb' Maxwell-Boltzmann halo provider on a bare v/c grid.

    The provider expects nu-velocity input and returns 1/(nu-velocity). To stay
    in bare v/c convention we multiply input by nu.c0 (lift to nu velocity)
    and the output by nu.c0 (convert 1/v_nu -> 1/(v/c)).

    Args:
        vmin_over_c: real ndarray, shape (N_q, N_E), vmin/c (dimensionless)
        halo_distribution: DM_Halo_Distributions instance (configured astro)
    Returns:
        real ndarray, shape (N_q, N_E), eta in c^-1 units.
    """
    vmin_nu_t = torch.as_tensor(vmin_over_c) * nu.c0
    eta_nu_t = halo_distribution.eta_MB_tensor(vmin_nu_t)
    return eta_nu_t.detach().cpu().numpy() * nu.c0


def _eta_scalar_bare(vmin_over_c, halo_fn):
    """Bridge scalar nu-velocity halo methods (etaSHM, etaTsa, etaDPL) to the bare v/c grid.

    Calls halo_fn element-wise. Slow (scipy quad per point) but correct.

    Args:
        vmin_over_c: real ndarray, shape (N_q, N_E), vmin/c (dimensionless)
        halo_fn: callable accepting a single nu-velocity and returning eta in 1/(nu-velocity)
    Returns:
        real ndarray, shape (N_q, N_E), eta in c^-1 units.
    """
    def _eval(v_over_c):
        return float(halo_fn(v_over_c * nu.c0)) * nu.c0
    return np.vectorize(_eval)(vmin_over_c)


def _eta_provider_bare(vmin_over_c, eta_provider):
    """Evaluate a legacy vectorized eta provider on a bare v/c grid."""
    vmin_t = torch.as_tensor(vmin_over_c, dtype=torch.float64) * nu.c0
    eta_nu = eta_provider(vmin_t)
    if not isinstance(eta_nu, torch.Tensor):
        eta_nu = torch.as_tensor(eta_nu, dtype=torch.float64)
    return eta_nu.detach().cpu().numpy() * nu.c0


def _file_eta_provider(halo_model, halo_distribution, rhoX_eV_per_cm3, sigma_e_cm2):
    """Build the same file-backed/interpolated halo provider used by DMeRate."""
    from DMeRates.halo.file_loader import FileHaloProvider, load_halo_file_data

    module_dir = Path(__file__).resolve().parents[1]
    rhoX_mass_density = rhoX_eV_per_cm3 * nu.eV / (nu.c0**2 * nu.cm**3)
    file_vmins, file_etas = load_halo_file_data(
        module_dir=module_dir,
        dm_halo=halo_distribution,
        halo_model=halo_model,
        v0=halo_distribution.v0,
        vEarth=halo_distribution.vEarth,
        vEscape=halo_distribution.vEscape,
        rhoX=rhoX_mass_density,
        cross_section=sigma_e_cm2 * nu.cm**2,
        default_dtype=torch.float64,
    )
    return FileHaloProvider(file_vmins, file_etas)


def _resolve_eta(halo_model, vmin_over_c, halo_distribution, eta_provider=None):
    """Dispatch to the right halo provider for a given model key."""
    if eta_provider is not None:
        return _eta_provider_bare(vmin_over_c, eta_provider)
    if halo_model == "imb":
        return _eta_imb_bare(vmin_over_c, halo_distribution)
    if halo_model == "shm":
        return _eta_scalar_bare(vmin_over_c, halo_distribution.etaSHM)
    if halo_model == "tsa":
        return _eta_scalar_bare(vmin_over_c, halo_distribution.etaTsa)
    if halo_model == "dpl":
        return _eta_scalar_bare(vmin_over_c, halo_distribution.etaDPL)
    raise NotImplementedError(
        f"halo_model={halo_model!r} is not yet supported by the QCDark2 engine. "
        "Supported: 'imb', 'shm', 'tsa', 'dpl'."
    )


# ----------------------------------------------------------------------------
# Engine — Si composite / heavy-mediator / MB / RPA validated case.
# ----------------------------------------------------------------------------

def _bare_floats_from_loader(d):
    """Convert dielectric_response (numericalunits-typed) fields to bare QCDark2 floats."""
    M_cell_eV = float(d.M_cell / nu.eV)             # cell rest energy in eV
    V_cell_bohr = float(d.V_cell_bohr)              # already raw Bohr^3
    q_ame = np.asarray(d.q_ame, dtype=np.float64)   # already raw alpha*m_e
    E_eV = np.asarray(d.E / nu.eV, dtype=np.float64)
    eps = np.asarray(d.epsilon, dtype=np.complex128)
    return eps, q_ame, E_eV, M_cell_eV, V_cell_bohr


def _qcdark2_constants_bare():
    """Reproduce QCDark2's hard-coded constants from numericalunits.

    Section 5 of the derivation notebook proves these match QCDark2's
    hard-coded `kg`, `alpha`, `m_e`, `c_kms`, `cm2sec`, `sec2yr` to <1e-7.
    """
    kg_eVoverc2 = nu.kg * nu.c0 ** 2 / nu.eV       # 1 kg in eV/c^2 (energy)
    alpha_FS = float(nu.alphaFS)
    me_eV = float(nu.me * nu.c0 ** 2 / nu.eV)      # m_e c^2 in eV
    c_kms = float(nu.c0 / (nu.km / nu.s))          # c in km/s
    cm2sec = 1.0 / c_kms * 1e-5                     # s/cm
    sec2yr = 1.0 / (60.0 * 60.0 * 24.0 * 365.25)
    return float(kg_eVoverc2), alpha_FS, me_eV, c_kms, cm2sec, sec2yr


@dataclass
class DielectricRateResult:
    """Bare-float view of the dR/dE result for QCDark2 engine validation.

    spectrum: a RateSpectrum (carries nu units; suitable for the wider DMeRates API)
    E_eV: numpy array of energies in eV (bare)
    dRdE_per_kg_per_year_per_eV: numpy array, the bare-float result
    """
    spectrum: RateSpectrum
    E_eV: np.ndarray
    dRdE_per_kg_per_year_per_eV: np.ndarray


def _qcdark2_half_open_mask(q, q_min, q_max):
    """Bool mask matching QCDark2's Python-slice q_i:q_f convention (upper bound excluded).

    Args:
        q:     (N_q,) tensor of momentum-transfer values in eV.
        q_min: (N_v, N_E) lower kinematic bound.
        q_max: (N_v, N_E) upper kinematic bound.
    Returns:
        (N_v, N_q, N_E) bool tensor.
    """
    q3 = q[None, :, None]
    mask_strict = (
        (q3 > q_min[:, None, :])
        & (q3 < q_max[:, None, :])
        & (q_max[:, None, :] > q_min[:, None, :])
    )
    mask_rev = torch.flip(mask_strict, dims=[1])
    cum_r = torch.cumsum(mask_rev.to(torch.int32), dim=1)
    last_true = torch.flip(mask_rev & (cum_r == 1), dims=[1])
    return mask_strict & ~last_true


# ----------------------------------------------------------------------------
# SRDM (solar-reflected DM) path — relativistic flux from external files.
# ----------------------------------------------------------------------------

def _compute_dRdE_srdm(
    *,
    material: str,
    mX_eV: float,
    sigma_e_cm2: float,
    FDMn: int,
    mediator_spin: str,
    screening,
    variant: str = "composite",
    dielectric=None,
) -> "DielectricRateResult":
    """Native QCDark2 SRDM dR/dE.

    Required validated case (Phase B):
        material='Si', mX_eV=48232.9466, sigma_e_cm2=1.098541e-38, FDMn=2,
        mediator_spin='vector', halo_model='srdm', screening='rpa',
        variant='composite'.
    """
    screening = normalize_dielectric_screening(screening)

    mediator_spin = _normalize_mediator_spin(mediator_spin)
    flux_spin = _flux_mediator_spin(mediator_spin)

    if FDMn not in _VALID_MEDIATORS:
        raise ValueError(
            f"Unsupported FDMn={FDMn}. Supported: {sorted(_VALID_MEDIATORS)}"
        )

    # ---- Dielectric data.
    if dielectric is None:
        dielectric = dielectric_response(material, variant=variant)
    eps, q_ame, E_eV, M_cell_eV, V_cell_bohr = _bare_floats_from_loader(dielectric)

    # ---- Bare constants.
    kg_QCD, alpha_FS, me_eV, _c_kms, _cm2sec, sec2yr = _qcdark2_constants_bare()
    ame_eV = alpha_FS * me_eV
    q_eV = q_ame * ame_eV

    # ---- Flux (via SRDM infrastructure).
    # load_srdm_flux is the only km/s -> v/c conversion boundary. It returns
    # dPhi/d(v/c) in numericalunits, so the engine integrates that directly over
    # dimensionless v/c. Do not convert back to raw dPhi/dv_kms here; that
    # convention is only for QCDark2 reference-code matching in the notebook.
    from DMeRates.srdm.flux_loader import load_srdm_flux as _load_flux
    from DMeRates.srdm.manifest import find_entry as _find_entry
    from DMeRates.data.registry import DataRegistry as _DR

    v_tensor, dphi_tensor = _load_flux(mX_eV, sigma_e_cm2, FDMn, mediator_spin)
    entry = _find_entry(mX_eV, sigma_e_cm2, FDMn, mediator_spin)
    _resolved_flux_path = _DR.srdm_flux_file(entry["filename"])

    v_oc = v_tensor.numpy()
    dphi_dv_over_c = dphi_tensor.numpy() * float(nu.cm**2 * nu.s)

    # ---- Material constants.
    V_cell_eV3 = V_cell_bohr / ame_eV**3
    N_cell = 2
    n_density = N_cell / V_cell_eV3
    mu_chi_e = me_eV * mX_eV / (me_eV + mX_eV)
    mA_eV_val = 0.0 if FDMn == 2 else _LARGE_MA_EV

    # ---- ELF and screening.
    elf = energy_loss_function(eps)
    screen_ratio = dielectric_screening_ratio(eps, screening)
    elf_eff = elf * screen_ratio

    # ---- Vectorized SRDM integrand (torch, CPU).
    # Peak tensor shape: (N_v, N_q, N_E) ≈ (299, 1251, 501).
    from DMeRates.srdm.kinematics import (
        q_bounds as _q_bounds,
        srdm_integrand_kernel as _srdm_integrand_kernel,
        reference_propagator_factor as _ref_prop,
    )

    v = torch.as_tensor(v_oc, dtype=torch.float64)
    q = torch.as_tensor(q_eV, dtype=torch.float64)
    E = torch.as_tensor(E_eV, dtype=torch.float64)
    elf_t = torch.as_tensor(elf_eff, dtype=torch.float64)
    phi = torch.as_tensor(dphi_dv_over_c, dtype=torch.float64)

    gamma_v = 1.0 / torch.sqrt(1.0 - v**2)
    E_chi = gamma_v * mX_eV

    E_chi_3 = E_chi[:, None, None]
    q3 = q[None, :, None]
    E3 = E[None, None, :]

    kernel = _srdm_integrand_kernel(
        q3,
        E3,
        E_chi_3,
        mX_eV,
        mA_eV_val,
        mediator_spin,
    )
    integrand_q = kernel * elf_t[None, :, :]

    # ---- q-integration: masked trapezoid matching QCDark2 slicing convention.
    q_min, q_max = _q_bounds(v, E, mX_eV)
    mask_open = _qcdark2_half_open_mask(q, q_min, q_max)

    dq = q[1:] - q[:-1]
    bin_contrib = 0.5 * (integrand_q[:, :-1, :] + integrand_q[:, 1:, :]) * dq[None, :, None]
    bin_valid = mask_open[:, :-1, :] & mask_open[:, 1:, :]
    sigma_per_v = (bin_contrib * bin_valid).sum(dim=1)

    # ---- Per-velocity prefactor (eq. A.21).
    ref_prop = _ref_prop(mA_eV_val, alpha_FS, me_eV)
    prefactor_v = (
        sigma_e_cm2
        / (32.0 * np.pi**2 * alpha_FS * v**2 * E_chi)
        * ref_prop
        / mu_chi_e**2
        / n_density
    )
    sigma_per_v = sigma_per_v * prefactor_v[:, None]

    # ---- v-integration via trapezoid; convert to events/kg/year/eV.
    dR = torch.trapezoid(sigma_per_v * phi[:, None], v, dim=0).numpy()
    dRdE_bare = (N_cell / M_cell_eV) * dR * kg_QCD / sec2yr

    # ---- Wrap in RateSpectrum.
    E_t = torch.as_tensor(E_eV, dtype=torch.float64) * nu.eV
    dRdE_t = torch.as_tensor(dRdE_bare, dtype=torch.float64) / (
        nu.kg * nu.year * nu.eV
    )
    spectrum = RateSpectrum(
        E=E_t,
        dR_dE=dRdE_t,
        material=material,
        backend="qcdark2",
        metadata=dict(
            halo_model="srdm",
            mediator_spin=mediator_spin,
            flux_mediator_spin=flux_spin,
            flux_file=str(_resolved_flux_path),
            mX_eV=float(mX_eV),
            sigma_e_cm2=float(sigma_e_cm2),
            FDMn=int(FDMn),
            screening=screening,
            variant=variant,
        ),
    )
    return DielectricRateResult(
        spectrum=spectrum,
        E_eV=E_eV,
        dRdE_per_kg_per_year_per_eV=dRdE_bare,
    )


def compute_dRdE(
    *,
    material: str,
    mX_eV: float,
    FDMn: int,
    halo_model: str,
    screening,                      # required keyword; explicit None or omission raises
    variant: str = "composite",
    sigma_e_cm2: float = 1e-39,
    rhoX_eV_per_cm3: float = 0.3e9,
    halo_distribution=None,         # DM_Halo_Distributions instance (default: SHM defaults)
    eta_provider=None,              # optional vectorized eta(v_min_nu) provider
    dielectric=None,                # optional pre-loaded dielectric_response
    mediator_spin: str = "vector",  # required for SRDM; ignored for halo models
):
    """Native QCDark2 dR/dE.

    Required validated case (Step 3 of the runbook):
        material='Si', variant='composite', mX_eV=1e9, FDMn=0,
        halo_model='imb', screening='rpa'.

    Args:
        material: material key understood by `dielectric_response`.
        mX_eV: dark matter mass in eV.
        FDMn: mediator-power exponent. 0 = heavy (F_DM=1); 2 = light (F_DM = (alpha m_e/q)^2).
        halo_model: DMeRates halo key. Supported: 'imb', 'shm', 'tsa', 'dpl'.
            'imb' uses the fast tensor Maxwell-Boltzmann path. 'shm', 'tsa',
            and 'dpl' use a vectorized file-backed/interpolated eta provider
            when possible.
        screening: 'rpa' (use the loaded full epsilon, ratio is identically 1)
                   or 'none' (set epsilon_screen = 1 -> include |epsilon|^2 factor).
                   None or omission raises ValueError.
        variant: QCDark2 dielectric variant ('composite', 'lfe', 'nolfe').
        sigma_e_cm2: bar-sigma_e in cm^2.
        rhoX_eV_per_cm3: local DM ENERGY density in eV/cm^3 (note: NOT mass density).
        halo_distribution: optional `DM_Halo_Distributions` instance with custom
            v0, vEarth, vEscape. Defaults to one constructed from Constants.py
            defaults (v0=238 km/s, vEarth=250.2 km/s, vEscape=544 km/s).
        eta_provider: optional callable accepting nu-velocity tensor v_min and
            returning eta in 1/(nu-velocity). Used by the legacy DMeRate path to
            reuse file-backed interpolation and halo-independent providers.
        dielectric: optional pre-loaded `dielectric_response` (saves an HDF5 read
            on repeated calls). When `None`, one is constructed from material/variant.

    Returns:
        DielectricRateResult with `.E_eV`, `.dRdE_per_kg_per_year_per_eV`,
        and a `.spectrum: RateSpectrum` ready for downstream DMeRates code.
    """
    # ---- SRDM dispatch (solar-reflected DM uses external flux, not halo eta).
    if halo_model == "srdm":
        return _compute_dRdE_srdm(
            material=material,
            mX_eV=mX_eV,
            sigma_e_cm2=sigma_e_cm2,
            FDMn=FDMn,
            mediator_spin=mediator_spin,
            screening=screening,
            variant=variant,
            dielectric=dielectric,
        )

    # ---- Required-screening guard and normalization (Step 3.4 policy).
    screening = normalize_dielectric_screening(screening)

    if FDMn not in _VALID_MEDIATORS:
        raise ValueError(
            f"Unsupported FDMn={FDMn}. Supported: {sorted(_VALID_MEDIATORS)}"
        )

    # ---- Halo provider.
    if halo_distribution is None:
        from DMeRates.DM_Halo import DM_Halo_Distributions
        halo_distribution = DM_Halo_Distributions()
    if eta_provider is None and halo_model in {"shm", "tsa", "dpl"}:
        eta_provider = _file_eta_provider(
            halo_model=halo_model,
            halo_distribution=halo_distribution,
            rhoX_eV_per_cm3=rhoX_eV_per_cm3,
            sigma_e_cm2=sigma_e_cm2,
        ).eta

    # ---- Dielectric data.
    if dielectric is None:
        dielectric = dielectric_response(material, variant=variant)
    eps, q_ame, E_eV, M_cell_eV, V_cell_bohr = _bare_floats_from_loader(dielectric)

    # ---- Reproduced QCDark2 constants (bare).
    kg_QCD, alpha_FS, me_eV, c_kms, cm2sec, sec2yr = _qcdark2_constants_bare()

    # ---- Scalar quantities (bare).
    rho_T = M_cell_eV / kg_QCD / V_cell_bohr           # kg / Bohr^3
    mu = mX_eV * me_eV / (mX_eV + me_eV)               # eV
    prefactor = (
        (1.0 / rho_T)
        * (rhoX_eV_per_cm3 / mX_eV)
        * (sigma_e_cm2 / mu ** 2)
        / (4.0 * np.pi)
    )

    # ---- Dynamic structure factor S(q,E) and mediator F_DM^2.
    S = dynamic_structure_factor(eps, q_ame, alpha=alpha_FS)
    F2 = mediator_factor(q_ame, FDMn)

    # ---- Screening ratio |eps|^2 / |eps_screen|^2.
    screen_ratio = dielectric_screening_ratio(eps, screening)

    # ---- v_min and eta (bare, in v/c convention).
    q_eV = q_ame * alpha_FS * me_eV
    vmin_bare = v_min_bare(q_eV, E_eV, mX_eV)            # (N_q, N_E)
    eta = _resolve_eta(
        halo_model,
        vmin_bare,
        halo_distribution,
        eta_provider=eta_provider,
    )  # (N_q, N_E), c^-1

    # ---- Integrand and q-integral (Simpson over q in eV).
    integrand = q_ame[:, None] * F2[:, None] * S * eta * screen_ratio  # (N_q, N_E)
    dRdE_inner = _spi.simpson(integrand, x=q_eV, axis=0)               # (N_E,)

    # ---- Final assembly into events / kg / year / eV.
    dRdE_bare = prefactor * dRdE_inner / cm2sec / sec2yr

    # ---- Pack as RateSpectrum (E carries nu units; dR_dE is bare * (events/kg/yr/eV) target unit).
    E_t = torch.as_tensor(E_eV, dtype=torch.float64) * nu.eV
    dRdE_t = torch.as_tensor(dRdE_bare, dtype=torch.float64) / (nu.kg * nu.year * nu.eV)
    spectrum = RateSpectrum(
        E=E_t,
        dR_dE=dRdE_t,
        material=material,
        backend="qcdark2",
        metadata=dict(
            variant=variant,
            mX_eV=float(mX_eV),
            FDMn=int(FDMn),
            halo_model=halo_model,
            screening=screening,
            sigma_e_cm2=float(sigma_e_cm2),
            rhoX_eV_per_cm3=float(rhoX_eV_per_cm3),
            v0_kms=float(halo_distribution.v0 / (nu.km / nu.s)),
            vEarth_kms=float(halo_distribution.vEarth / (nu.km / nu.s)),
            vEscape_kms=float(halo_distribution.vEscape / (nu.km / nu.s)),
        ),
    )
    return DielectricRateResult(
        spectrum=spectrum,
        E_eV=E_eV,
        dRdE_per_kg_per_year_per_eV=dRdE_bare,
    )


__all__ = [
    "energy_loss_function",
    "dynamic_structure_factor",
    "mediator_factor",
    "v_min_bare",
    "compute_dRdE",
    "DielectricRateResult",
]
