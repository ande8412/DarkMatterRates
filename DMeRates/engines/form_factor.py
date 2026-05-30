import numpy as np
import numericalunits as nu
import torch

from DMeRates.screening.lindhard import DEFAULT_LINDHARD_ETA_EV
from DMeRates.screening.semiconductor import (
    normalize_semiconductor_screening,
    semiconductor_native_screening_factor,
    semiconductor_screening_factor,
)
from DMeRates.spectrum import RateSpectrum
from DMeRates.srdm.mediators import (
    flux_mediator_spin as _flux_mediator_spin,
    normalize_mediator_spin as _normalize_mediator_spin,
)

# Legacy QEDark convention: grid values summed directly without dE bin-width factor.
# Equivalent to dE * 10 for the current 0.1 eV grid. See tests/phase0_2.md.
_LEGACY_QEDARK_ENERGY_NORM = 1.0 * nu.eV
_VALID_MEDIATORS = {0, 2}
_QEDARK_LEGACY_WK = 2.0 / 137.0
_SRDM_HALO_MODELS = {"srdm", "srdm_modulated"}


def _FORM_FACTOR_TO_S_PREFACTOR(q_eV, V_cell_eV3, *, alpha_FS, me_eV):
    """Named f^2 -> S prefactor from tests/qcdark2_srdm_derivation.ipynb, Section 6.

    S(omega, q) = prefactor(q) * f^2(q, omega), with q in eV and V_cell in eV^-3.
    """
    return 8.0 * np.pi**2 * (alpha_FS * me_eV) ** 2 / (
        V_cell_eV3 * q_eV[:, None] ** 3
    )


def _form_factor_to_elf_equivalent(f2, q_eV, V_cell_eV3, *, alpha_FS, me_eV):
    """Convert f^2(q, omega) to the effective response used by the SRDM kernel.

    The reused QCDark2-style SRDM kernel consumes an ELF-shaped response. For
    crystal form factors, matching the solar-reflection crystal-rate formula in
    the nonrelativistic limit requires S(q, E) / q^2 here. Adding the dielectric
    identity factor 2*pi*alpha would suppress the form-factor rate by 2*pi*alpha.
    """
    S = _FORM_FACTOR_TO_S_PREFACTOR(
        q_eV,
        V_cell_eV3,
        alpha_FS=alpha_FS,
        me_eV=me_eV,
    ) * f2
    return S / q_eV[:, None] ** 2


def _qedark_cell_constants(material):
    """QEDark tables carry f^2 only; use the matching QCDark1 cell constants."""
    import h5py

    from DMeRates.data.registry import DataRegistry

    with h5py.File(DataRegistry.qcdark1_ff(material), "r") as h5:
        V_cell_eV3 = float(h5["results"].attrs["VCell"])
        M_cell_eV = float(h5["results"].attrs["mCell"])
    return V_cell_eV3, M_cell_eV


def _form_factor_bare_constants():
    """Bare alpha and electron rest energy, matching the SRDM notebook convention."""
    return float(nu.alphaFS), float(nu.me * nu.c0**2 / nu.eV)


def _form_factor_srdm_grids(*, backend, material, form_factor):
    """Return bare-float q, E, f^2, V_cell, and M_cell for a form-factor backend."""
    if form_factor is None:
        from DMeRates.data.registry import DataRegistry
        from DMeRates.responses.qcdark1 import form_factor as qcdark1_form_factor
        from DMeRates.responses.qedark import form_factorQEDark

        if backend == "qcdark1":
            form_factor = qcdark1_form_factor(str(DataRegistry.qcdark1_ff(material)))
        elif backend == "qedark":
            form_factor = form_factorQEDark(str(DataRegistry.qedark_ff(material)))

    if backend == "qcdark1":
        f2 = np.asarray(form_factor.ff, dtype=np.float64)
        n_q, n_E = f2.shape
        dq_ame = float(form_factor.dq / (nu.alphaFS * nu.me * nu.c0))
        dE_eV = float(form_factor.dE / nu.eV)
        alpha_FS, me_eV = _form_factor_bare_constants()
        q_eV = (np.arange(n_q, dtype=np.float64) + 0.5) * dq_ame * alpha_FS * me_eV
        E_eV = (np.arange(n_E, dtype=np.float64) + 0.5) * dE_eV
        V_cell_eV3 = float(form_factor.VCell / nu.eV)
        M_cell_eV = float(form_factor.mCell * nu.c0**2 / nu.eV)
        return q_eV, E_eV, f2, V_cell_eV3, M_cell_eV

    if backend == "qedark":
        # DMeRates' legacy halo loader stores QEDark as (wk / 4) * f_crys.
        # The SRDM f^2 -> S derivation in tests/qcdark2_srdm_derivation.ipynb
        # uses the raw tabulated f_crys representation, so undo only that legacy
        # halo normalization here. The halo path remains unchanged.
        f2 = np.asarray(form_factor.ff, dtype=np.float64) * (4.0 / _QEDARK_LEGACY_WK)
        n_q, n_E = f2.shape
        alpha_FS, me_eV = _form_factor_bare_constants()
        dq_ame = float(form_factor.dq / (nu.alphaFS * nu.me * nu.c0))
        dE_eV = float(form_factor.dE / nu.eV)
        q_eV = np.arange(1, n_q + 1, dtype=np.float64) * dq_ame * alpha_FS * me_eV
        E_eV = np.arange(1, n_E + 1, dtype=np.float64) * dE_eV
        V_cell_eV3, M_cell_eV = _qedark_cell_constants(material)
        return q_eV, E_eV, f2, V_cell_eV3, M_cell_eV

    raise ValueError(f"Unsupported form-factor SRDM backend: {backend!r}")


def _compute_dRdE_srdm_form_factor(
    *,
    backend: str,
    material: str,
    mX_eV: float,
    sigma_e_cm2: float,
    FDMn: int,
    mediator_spin: str,
    DoScreen: bool,
    form_factor=None,
    screening: str | None = None,
    lindhard_eta_eV: float = DEFAULT_LINDHARD_ETA_EV,
    halo_model: str = "srdm",
    flux_source=None,
    ring_index: int | None = None,
    srdm_base_data_dir=None,
) -> RateSpectrum:
    """SRDM rate from a crystal form factor f^2(q, omega).

    Steps:
        1. Load f^2(q, omega), q, E, V_cell from the form-factor representation.
        2. Build S(omega, q) = 8 pi^2 (alpha m_e)^2 / V_cell * f^2 / q^3.
        3. Apply selected 1 / |eps|^2 screening on the backend's native q/E grid.
        4. Reuse the dielectric SRDM kinematics and flux-integration kernel.
        5. Return a RateSpectrum with SRDM metadata.
    """
    mediator_spin = _normalize_mediator_spin(mediator_spin)
    flux_spin = _flux_mediator_spin(mediator_spin)

    if FDMn not in _VALID_MEDIATORS:
        raise ValueError(
            f"Unsupported FDMn={FDMn}. Supported: {sorted(_VALID_MEDIATORS)}"
        )

    from DMeRates.engines.dielectric import (
        _LARGE_MA_EV,
        _qcdark2_constants_bare,
        _qcdark2_half_open_mask,
    )
    from DMeRates.srdm.flux_loader import resolve_srdm_flux_source
    from DMeRates.srdm.kinematics import (
        q_bounds,
        reference_propagator_factor,
        srdm_integrand_kernel,
    )

    v_tensor, dphi_tensor, flux_metadata = resolve_srdm_flux_source(
        source=flux_source,
        mX_MeV=float(mX_eV) / 1.0e6,
        mX_eV=mX_eV,
        sigma_e_cm2=sigma_e_cm2,
        FDMn=FDMn,
        mediator_spin=mediator_spin,
        ring_index=ring_index,
        base_data_dir=srdm_base_data_dir,
    )

    q_eV, E_eV, f2, V_cell_eV3, M_cell_eV = _form_factor_srdm_grids(
        backend=backend,
        material=material,
        form_factor=form_factor,
    )

    kg_QCD, alpha_FS, me_eV, _c_kms, _cm2sec, sec2yr = _qcdark2_constants_bare()
    elf_equiv = _form_factor_to_elf_equivalent(
        f2,
        q_eV,
        V_cell_eV3,
        alpha_FS=alpha_FS,
        me_eV=me_eV,
    )

    screening_mode = normalize_semiconductor_screening(screening, DoScreen)
    if screening_mode != "none":
        # The form-factor grids are native to each backend. Semiconductor
        # screening is evaluated on that same grid and composes multiplicatively
        # with S, matching the halo-path convention and the derivation notebook.
        q_t = torch.as_tensor(q_eV, dtype=torch.float64) * (nu.eV / nu.c0)
        E_t = torch.as_tensor(E_eV, dtype=torch.float64) * nu.eV
        screen = semiconductor_screening_factor(
            material,
            q_t,
            E_t,
            screening=screening_mode,
            do_screen=True,
            lindhard_eta_eV=lindhard_eta_eV,
        )
        elf_equiv = elf_equiv * screen.detach().cpu().numpy() ** 2

    v = v_tensor.to(dtype=torch.float64)
    phi = dphi_tensor.to(dtype=torch.float64) * float(nu.cm**2 * nu.s)
    q = torch.as_tensor(q_eV, dtype=torch.float64)
    E = torch.as_tensor(E_eV, dtype=torch.float64)
    elf_t = torch.as_tensor(elf_equiv, dtype=torch.float64)

    N_cell = 2
    n_density = N_cell / V_cell_eV3
    mu_chi_e = me_eV * mX_eV / (me_eV + mX_eV)
    mA_eV = 0.0 if FDMn == 2 else _LARGE_MA_EV

    gamma_v = 1.0 / torch.sqrt(1.0 - v**2)
    E_chi = gamma_v * mX_eV
    E_chi_3 = E_chi[:, None, None]
    q3 = q[None, :, None]
    E3 = E[None, None, :]

    # Peak tensor shape is (N_v, N_q, N_E): QEDark/QCDark1 Si is
    # approximately (299, 900, 500), integrated in torch without v/q/E loops.
    kernel = srdm_integrand_kernel(
        q3,
        E3,
        E_chi_3,
        mX_eV,
        mA_eV,
        mediator_spin,
    )
    integrand_q = kernel * elf_t[None, :, :]

    q_min, q_max = q_bounds(v, E, mX_eV)
    mask_open = _qcdark2_half_open_mask(q, q_min, q_max)

    dq = q[1:] - q[:-1]
    bin_contrib = 0.5 * (integrand_q[:, :-1, :] + integrand_q[:, 1:, :]) * dq[None, :, None]
    bin_valid = mask_open[:, :-1, :] & mask_open[:, 1:, :]
    sigma_per_v = (bin_contrib * bin_valid).sum(dim=1)

    ref_prop = reference_propagator_factor(mA_eV, alpha_FS, me_eV)
    prefactor_v = (
        sigma_e_cm2
        / (32.0 * np.pi**2 * alpha_FS * v**2 * E_chi)
        * ref_prop
        / mu_chi_e**2
        / n_density
    )
    sigma_per_v = sigma_per_v * prefactor_v[:, None]

    dR = torch.trapezoid(sigma_per_v * phi[:, None], v, dim=0)
    dRdE_bare = (N_cell / M_cell_eV) * dR * kg_QCD / sec2yr

    E_t = torch.as_tensor(E_eV, dtype=torch.float64) * nu.eV
    dRdE_t = dRdE_bare / (nu.kg * nu.year * nu.eV)
    return RateSpectrum(
        E=E_t,
        dR_dE=dRdE_t,
        material=material,
        backend=backend,
        metadata=dict(
            halo_model=halo_model,
            mediator_spin=mediator_spin,
            flux_mediator_spin=flux_spin,
            flux_file=flux_metadata.get("flux_file"),
            flux_source=flux_metadata.get("flux_source"),
            ring_index=flux_metadata.get("ring_index"),
            ring_count=flux_metadata.get("ring_count"),
            mX_eV=float(mX_eV),
            sigma_e_cm2=float(sigma_e_cm2),
            FDMn=int(FDMn),
            DoScreen=bool(DoScreen),
            screening=screening_mode,
            lindhard_eta_eV=float(lindhard_eta_eV) if screening_mode == "lindhard" else None,
            grid_shape=(int(len(q_eV)), int(len(E_eV))),
        ),
    )


def semiconductor_dRdE_spectrum(
    *,
    material: str,
    mX,
    FDMn: int,
    halo_model: str,
    DoScreen: bool,
    integrate: bool,
    QEDark: bool,
    form_factor,
    qArr: torch.Tensor,
    Earr: torch.Tensor,
    Ei_array: torch.Tensor,
    dtype_str: str,
    reduced_mass_fn,
    fdm_fn,
    get_parametrized_eta_fn,
    vmin_tensor_fn,
    tfscreening_fn,
    thomas_fermi_screening_fn,
    halo_id_params=None,
    sigma_e_cm2: float = 1e-38,
    mediator_spin: str = "vector",
    screening: str | None = None,
    lindhard_eta_eV: float = DEFAULT_LINDHARD_ETA_EV,
    ring_index: int | None = None,
    modulated_source: str | None = None,
    srdm_base_data_dir=None,
):
    """Extracted semiconductor differential-rate engine for QEDark/QCDark1."""
    if halo_model in _SRDM_HALO_MODELS:
        backend = "qedark" if QEDark else "qcdark1"
        return _compute_dRdE_srdm_form_factor(
            backend=backend,
            material=material,
            mX_eV=float(mX) * 1.0e6,
            sigma_e_cm2=sigma_e_cm2,
            FDMn=FDMn,
            mediator_spin=mediator_spin,
            DoScreen=DoScreen,
            form_factor=form_factor,
            screening=screening,
            lindhard_eta_eV=lindhard_eta_eV,
            halo_model=halo_model,
            flux_source=modulated_source if halo_model == "srdm_modulated" else None,
            ring_index=ring_index,
            srdm_base_data_dir=srdm_base_data_dir,
        )

    mX = mX * nu.MeV / nu.c0**2
    rm = reduced_mass_fn(mX, nu.me)
    prefactor = nu.alphaFS * ((nu.me / rm) ** 2) * (1 / form_factor.mCell)
    ff_arr = torch.tensor(form_factor.ff, dtype=torch.get_default_dtype())

    if integrate:
        import torchquad
        from torchquad import Simpson, set_up_backend

        torchquad.set_log_level("ERROR")
        set_up_backend("torch", data_type=dtype_str)
        simp = Simpson()
        numq = len(qArr)
        qmin = qArr[0]
        qmax = qArr[-1]
        integration_domain = torch.tensor([[qmin, qmax]], dtype=torch.get_default_dtype())

        def vmin(q, E, mX_):
            term1 = E.unsqueeze(0) / q.unsqueeze(1)
            term2 = q.unsqueeze(1) / (2 * mX_)
            return term1 + term2

        def eta_func(vMin):
            return get_parametrized_eta_fn(vMin, mX, halo_model, halo_id_params=halo_id_params)

        def momentum_integrand(q):
            q = q.flatten()
            qdenom = 1 / q**2
            qdenom *= (fdm_fn(q, FDMn)) ** 2
            eta = eta_func(vmin(q, Earr, mX))
            tf_f = semiconductor_screening_factor(
                material,
                q,
                Earr,
                screening=screening,
                do_screen=DoScreen,
                lindhard_eta_eV=lindhard_eta_eV,
            ) ** 2
            ff_f = ff_arr[:-1, :]
            result = eta * tf_f * ff_f
            result = torch.einsum("i,ji->ji", Earr, result)
            result = torch.einsum("j,ji->ji", qdenom, result)
            return result

        integrated_result = (
            simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain)
            / Earr
        )
    else:
        fdm_factor = (fdm_fn(qArr, FDMn)) ** 2
        vMins = vmin_tensor_fn(qArr, Earr, mX)
        etas = get_parametrized_eta_fn(vMins, mX, halo_model, halo_id_params=halo_id_params)
        if QEDark:
            ff_arr = ff_arr[:, Ei_array - 1]
        ff_arr = ff_arr.T
        tf_factor = semiconductor_native_screening_factor(
            material,
            Earr,
            qArr,
            screening=screening,
            do_screen=DoScreen,
            lindhard_eta_eV=lindhard_eta_eV,
        ) ** 2
        result = torch.einsum("i,ij->ij", Earr, torch.ones_like(etas))
        result *= etas
        result *= fdm_factor
        result *= ff_arr
        result *= tf_factor
        qdenom = 1 / qArr
        result = torch.einsum("j,ij->ij", qdenom, result)
        integrated_result = torch.sum(result, axis=1) / Earr

    integrated_result *= prefactor
    integrated_result /= nu.c0
    band_gap_result = torch.where(Earr < form_factor.band_gap, 0, integrated_result)

    backend = "qedark" if QEDark else "qcdark1"
    screening_mode = normalize_semiconductor_screening(screening, DoScreen)
    return RateSpectrum(
        E=Earr,
        dR_dE=band_gap_result,
        material=material,
        backend=backend,
        metadata={
            "integrate": integrate,
            "DoScreen": DoScreen,
            "screening": screening_mode,
            "lindhard_eta_eV": float(lindhard_eta_eV) if screening_mode == "lindhard" else None,
            "FDMn": FDMn,
        },
    )


def semiconductor_rates_from_spectrum(
    spectrum: RateSpectrum,
    prob_fn_tiled: torch.Tensor,
    *,
    integrate: bool,
) -> torch.Tensor:
    """Convert semiconductor dR/dE spectra into dR/dn_e for requested bins."""
    if integrate:
        return torch.trapezoid(spectrum.dR_dE * prob_fn_tiled, x=spectrum.E, axis=1)
    return torch.sum(spectrum.dR_dE * prob_fn_tiled * _LEGACY_QEDARK_ENERGY_NORM, axis=1)
