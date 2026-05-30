import numericalunits as nu
import torch

from DMeRates.Constants import skip_keys
from DMeRates.spectrum import RateSpectrum
from DMeRates.srdm.mediators import normalize_mediator_spin


def noble_rate_dme_shell(
    *,
    mX,
    FDMn: int,
    halo_model: str,
    shell_key: str,
    form_factor,
    qArrdict: dict,
    Earr: torch.Tensor,
    dtype_str: str,
    reduced_mass_fn,
    fdm_fn,
    vmin_tensor_fn,
    get_parametrized_eta_fn,
    halo_id_params=None,
):
    """Extracted noble-gas shell engine from legacy rate_dme_shell()."""
    qArr = qArrdict[shell_key]
    qmax = qArr[-1]
    numq = len(qArr)
    rm = reduced_mass_fn(mX, nu.me)
    prefactor = 1 / (8 * form_factor.mCell * (rm) ** 2)
    prefactor /= nu.c0**2

    fdm_factor = (fdm_fn(qArr, FDMn)) ** 2
    vMins = vmin_tensor_fn(qArr, Earr, mX, shell_key)
    etas = get_parametrized_eta_fn(vMins, mX, halo_model, halo_id_params=halo_id_params)
    ff_arr = form_factor.ff[shell_key]
    result = torch.einsum("j,ij->ij", fdm_factor, etas)
    result *= ff_arr

    import torchquad
    from torchquad import Simpson, set_up_backend

    torchquad.set_log_level("ERROR")
    set_up_backend("torch", data_type=dtype_str)
    simp = Simpson()
    integration_domain = torch.Tensor([[0, qmax]])

    def momentum_integrand(q):
        qint = q.flatten()
        return torch.einsum("j,ij->ji", qint, result)

    integrated_result = (
        simp.integrate(momentum_integrand, dim=1, N=numq, integration_domain=integration_domain)
        / Earr
    )
    integrated_result *= prefactor
    return integrated_result


def _srdm_flux_tail_grid(v_over_c: torch.Tensor, dphi_dv: torch.Tensor):
    """Return cumulative ∫_v^∞ dβ (dPhi/dβ) / β² on the SRDM flux grid."""
    order = torch.argsort(v_over_c)
    v = v_over_c[order]
    phi = dphi_dv[order]
    integrand = phi / v**2

    intervals = 0.5 * (integrand[:-1] + integrand[1:]) * (v[1:] - v[:-1])
    tail = torch.zeros_like(v)
    tail[:-1] = torch.flip(torch.cumsum(torch.flip(intervals, dims=[0]), dim=0), dims=[0])
    return v, integrand, tail


def _interp_srdm_flux_tail(
    vmin_over_c: torch.Tensor,
    v_grid: torch.Tensor,
    integrand_grid: torch.Tensor,
    tail_grid: torch.Tensor,
) -> torch.Tensor:
    """Interpolate the SRDM cumulative flux tail at arbitrary v_min/c values."""
    x = vmin_over_c.to(dtype=v_grid.dtype, device=v_grid.device)
    flat = x.reshape(-1)
    idx = torch.searchsorted(v_grid, flat, right=True) - 1
    idx = torch.clamp(idx, 0, len(v_grid) - 2)

    v0 = v_grid[idx]
    v1 = v_grid[idx + 1]
    g0 = integrand_grid[idx]
    g1 = integrand_grid[idx + 1]
    frac = (flat - v0) / (v1 - v0)
    gx = g0 + frac * (g1 - g0)
    partial = 0.5 * (g0 + gx) * (flat - v0)
    values = tail_grid[idx] - partial

    values = torch.where(flat <= v_grid[0], tail_grid[0], values)
    values = torch.where(flat >= v_grid[-1], torch.zeros_like(values), values)
    values = torch.clamp(values, min=0.0)
    return values.reshape_as(x).to(device=vmin_over_c.device)


def noble_srdm_rate_dme_shell(
    *,
    mX,
    FDMn: int,
    shell_key: str,
    form_factor,
    qArrdict: dict,
    Earr: torch.Tensor,
    reduced_mass_fn,
    fdm_fn,
    vmin_tensor_fn,
    sigma_e,
    v_grid: torch.Tensor,
    integrand_grid: torch.Tensor,
    tail_grid: torch.Tensor,
):
    """SRDM noble-gas shell rate using the atomic-ionization flux-tail formula."""
    qArr = qArrdict[shell_key]
    rm = reduced_mass_fn(mX, nu.me)
    prefactor = 1 / (8 * form_factor.mCell * (rm) ** 2)
    prefactor /= nu.c0**2

    fdm_factor = (fdm_fn(qArr, FDMn)) ** 2
    vMins = vmin_tensor_fn(qArr, Earr, mX, shell_key)
    flux_tail = _interp_srdm_flux_tail(
        vMins / nu.c0,
        v_grid.to(device=qArr.device),
        integrand_grid.to(device=qArr.device),
        tail_grid.to(device=qArr.device),
    )
    scatter_factor = sigma_e * flux_tail
    ff_arr = form_factor.ff[shell_key]

    result = torch.einsum("j,ij->ij", fdm_factor, scatter_factor)
    result *= ff_arr
    result = torch.einsum("j,ij->ij", qArr, result)
    integrated_result = torch.trapezoid(result, x=qArr, dim=1) / Earr
    integrated_result *= prefactor
    return integrated_result


def noble_srdm_dRdE_spectrum(
    *,
    material: str,
    mX,
    FDMn: int,
    mediator_spin: str,
    sigma_e_cm2: float,
    form_factor,
    qArrdict: dict,
    Earr: torch.Tensor,
    reduced_mass_fn,
    fdm_fn,
    vmin_tensor_fn,
    halo_model: str = "srdm",
    modulated_source: str | None = None,
    ring_index: int | None = None,
    srdm_base_data_dir=None,
):
    """Return a shell-resolved noble-gas SRDM RateSpectrum for one mass point.

    This follows the atomic-ionization formula in arXiv:2404.10066 Eq. 49/50:
    the halo eta factor is replaced by ∫_{v>vmin} d(v/c) dPhi/d(v/c) / (v/c)^2.
    """
    mediator_spin = normalize_mediator_spin(mediator_spin)
    if mediator_spin != "vector":
        raise NotImplementedError(
            "Noble-gas SRDM currently supports mediator_spin='vector' only."
        )

    from DMeRates.srdm.flux_loader import resolve_srdm_flux_source

    mX_eV = float(mX) * 1.0e6
    v_flux, dphi_dv, flux_metadata = resolve_srdm_flux_source(
        source=modulated_source if halo_model == "srdm_modulated" else None,
        mX_MeV=float(mX),
        mX_eV=mX_eV,
        sigma_e_cm2=sigma_e_cm2,
        FDMn=FDMn,
        mediator_spin=mediator_spin,
        ring_index=ring_index,
        base_data_dir=srdm_base_data_dir,
    )
    v_grid, integrand_grid, tail_grid = _srdm_flux_tail_grid(v_flux, dphi_dv)

    mX_nu = mX * nu.MeV / nu.c0**2
    sigma_e = sigma_e_cm2 * nu.cm**2
    shell_spectra = {}
    for key in form_factor.keys:
        if key in skip_keys[material]:
            continue
        shell_spectra[key] = noble_srdm_rate_dme_shell(
            mX=mX_nu,
            FDMn=FDMn,
            shell_key=key,
            form_factor=form_factor,
            qArrdict=qArrdict,
            Earr=Earr,
            reduced_mass_fn=reduced_mass_fn,
            fdm_fn=fdm_fn,
            vmin_tensor_fn=vmin_tensor_fn,
            sigma_e=sigma_e,
            v_grid=v_grid,
            integrand_grid=integrand_grid,
            tail_grid=tail_grid,
        )

    summed = torch.sum(torch.stack(list(shell_spectra.values())), axis=0)
    return RateSpectrum(
        E=Earr,
        dR_dE=summed,
        material=material,
        backend="noble_gas",
        metadata={
            "halo_model": halo_model,
            "mediator_spin": mediator_spin,
            "flux_mediator_spin": "vector",
            "flux_file": flux_metadata.get("flux_file"),
            "flux_source": flux_metadata.get("flux_source"),
            "ring_index": flux_metadata.get("ring_index"),
            "ring_count": flux_metadata.get("ring_count"),
            "mX_eV": mX_eV,
            "sigma_e_cm2": float(sigma_e_cm2),
            "FDMn": int(FDMn),
            "grid_shape": (int(len(v_grid)), int(len(Earr))),
        },
        shell_spectra=shell_spectra,
        shell_labels=list(shell_spectra.keys()),
    )


def noble_dRdE_spectrum(
    *,
    material: str,
    mX,
    FDMn: int,
    halo_model: str,
    form_factor,
    qArrdict: dict,
    Earr: torch.Tensor,
    dtype_str: str,
    reduced_mass_fn,
    fdm_fn,
    vmin_tensor_fn,
    get_parametrized_eta_fn,
    halo_id_params=None,
):
    """Return a shell-resolved RateSpectrum for noble-gas targets."""
    mX = mX * nu.MeV / nu.c0**2
    shell_spectra = {}
    for key in form_factor.keys:
        if key in skip_keys[material]:
            continue
        shell_spectra[key] = noble_rate_dme_shell(
            mX=mX,
            FDMn=FDMn,
            halo_model=halo_model,
            shell_key=key,
            form_factor=form_factor,
            qArrdict=qArrdict,
            Earr=Earr,
            dtype_str=dtype_str,
            reduced_mass_fn=reduced_mass_fn,
            fdm_fn=fdm_fn,
            vmin_tensor_fn=vmin_tensor_fn,
            get_parametrized_eta_fn=get_parametrized_eta_fn,
            halo_id_params=halo_id_params,
        )
    summed = torch.sum(torch.stack(list(shell_spectra.values())), axis=0)
    return RateSpectrum(
        E=Earr,
        dR_dE=summed,
        material=material,
        backend="noble_gas",
        metadata={"FDMn": FDMn},
        shell_spectra=shell_spectra,
        shell_labels=list(shell_spectra.keys()),
    )
