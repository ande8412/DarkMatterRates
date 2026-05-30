"""QCDark2 dielectric response loader.

Pre-check (Si_comp.h5, 2026-04-24):
- q range: 0.010 to 25.010 (in alpha*me units)
- V_cell: 270.11 (Bohr^3)
- M_cell: 5.216e10 (eV)

Convention confirmed: q is stored in alpha*me momentum units, V_cell is in
Bohr^3, and M_cell is in eV.
"""

from pathlib import Path

import h5py
import numericalunits as nu
import numpy as np

from DMeRates.data.registry import DataRegistry

_VALID_VARIANTS = {"composite", "lfe", "nolfe"}
_VALID_MATERIALS = {"Si", "Ge", "GaAs", "SiC", "diamond", "Diamond"}


class dielectric_response:
    """Load QCDark2 dielectric response data with numericalunits conversions."""

    def __init__(self, material: str, variant: str = "composite", filename=None):
        if material not in _VALID_MATERIALS:
            raise ValueError(
                f"Unsupported QCDark2 material '{material}'. "
                "Use one of: Si, Ge, GaAs, SiC, diamond."
            )
        if variant not in _VALID_VARIANTS:
            raise ValueError(
                f"Unsupported QCDark2 variant '{variant}'. "
                "Use one of: composite, lfe, nolfe."
            )

        self.material = material
        self.variant = variant
        material_key = "diamond" if material in {"diamond", "Diamond"} else material

        if filename is None:
            path = DataRegistry.qcdark2_dielectric(material_key, variant)
        else:
            path = Path(filename)
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"QCDark2 dielectric file not found: {path}\n"
                "Run  python -m DMeRates fetch-data  to download the dielectric files.\n"
                "To fetch only this material/variant: "
                f"python -m DMeRates fetch-data -m {material} -v {variant}"
            )

        self.path = str(path)
        q_amu = nu.alphaFS * nu.me * nu.c0
        bohr = nu.hbar / (nu.alphaFS * nu.me * nu.c0)
        with h5py.File(path, "r") as h5:
            self.epsilon = h5["epsilon"][...].copy()
            self.q_ame = h5["q"][...].copy()
            self.q = self.q_ame * q_amu
            self.E = h5["E"][...].copy() * nu.eV
            self.M_cell = float(h5.attrs["M_cell"]) * nu.eV
            self.V_cell_bohr = float(h5.attrs["V_cell"])
            self.V_cell = self.V_cell_bohr * bohr**3
            if "dE" in h5.attrs:
                self.dE = float(h5.attrs["dE"]) * nu.eV
            else:
                self.dE = float(np.mean(np.diff(h5["E"][...]))) * nu.eV


# Keep a form-factor style alias for consistency with existing loaders.
formFactorDielectric = dielectric_response
