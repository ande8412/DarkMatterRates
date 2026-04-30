"""Thin facade for backend-selectable rate calculations."""

from __future__ import annotations

from DMeRates.DMeRate import DMeRate


class RateCalculator:
    """Backend-selectable calculator that currently wraps the legacy DMeRate API."""

    def __init__(self, material, backend="qcdark1", variant="composite", screening=None, device=None):
        backend_norm = backend.lower()
        self.backend = backend_norm
        self.variant = variant
        self.screening = screening

        if backend_norm == "qcdark2":
            self._legacy = DMeRate(material, form_factor_type="qcdark2", device=device)
        elif backend_norm in {"qcdark1", "qcdark"}:
            self._legacy = DMeRate(material, form_factor_type="qcdark", device=device)
        elif backend_norm == "qedark":
            self._legacy = DMeRate(material, form_factor_type="qedark", device=device)
        elif backend_norm in {"noble_gas", "wimprates"}:
            self._legacy = DMeRate(material, form_factor_type="wimprates", device=device)
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. "
                "Use one of: qcdark2, qcdark1, qedark, wimprates."
            )

    def calculate_rates(
        self,
        *,
        mX_array,
        halo_model,
        FDMn,
        ne,
        pair_energy=None,
        mediator_spin="vector",
        **kwargs,
    ):
        """Calculate ne rates using the configured backend."""
        screening = kwargs.pop("screening", self.screening)
        if self.backend == "qcdark2":
            return self._legacy.calculate_rates(
                mX_array=mX_array,
                halo_model=halo_model,
                FDMn=FDMn,
                ne=ne,
                mediator_spin=mediator_spin,
                screening=screening,
                variant=self.variant,
                pair_energy=pair_energy,
                **kwargs,
            )
        return self._legacy.calculate_rates(
            mX_array=mX_array,
            halo_model=halo_model,
            FDMn=FDMn,
            ne=ne,
            mediator_spin=mediator_spin,
            screening=screening,
            **kwargs,
        )

    def calculate_spectrum(self, *, mX, halo_model, FDMn):
        """Return dR/dE for backends that expose a spectrum API (currently qcdark2)."""
        if self.backend != "qcdark2":
            raise NotImplementedError("calculate_spectrum is currently only implemented for backend='qcdark2'.")
        return self._legacy.calculate_qcdark2_spectrum(
            mX=mX,
            halo_model=halo_model,
            FDMn=FDMn,
            screening=self.screening,
            variant=self.variant,
        )
