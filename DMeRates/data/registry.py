import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # DarkMatterRates/


def _default_qcdark2_root() -> Path:
    """Resolve QCDark2 data from env, sibling checkout, or optional bundled copy."""
    env_root = os.environ.get("DMERATES_QCDARK2_ROOT")
    if env_root:
        return Path(env_root)

    sibling_root = _REPO_ROOT.parent / "QCDark2"
    if (sibling_root / "dielectric_functions").is_dir():
        return sibling_root

    bundled_root = _REPO_ROOT / "form_factors" / "QCDark2"
    if (bundled_root / "dielectric_functions").is_dir():
        return bundled_root

    return sibling_root


class DataRegistry:
    """Centralized path management for DMeRates data files."""

    # --- External data roots (large; not bundled) ---
    form_factor_root: Path = _REPO_ROOT / "form_factors"
    halo_root: Path        = _REPO_ROOT / "halo_data"
    rate_output_root: Path = _REPO_ROOT / "DMeRates" / "Rates"

    # --- QCDark2 data root (external; override via env var) ---
    qcdark2_root: Path = _default_qcdark2_root()

    # --- Default destination for 'python -m DMeRates fetch-data' ---
    bundled_qcdark2_root: Path = _REPO_ROOT / "form_factors" / "QCDark2"

    # --- QCDark2 dielectric file paths ---
    @classmethod
    def qcdark2_dielectric(cls, material_key: str, variant: str) -> Path:
        """Return path to a QCDark2 HDF5 dielectric file.

        material_key: 'Si', 'Ge', 'GaAs', 'SiC', 'Diamond' or 'diamond'
        variant: 'composite', 'lfe', 'nolfe'
        """
        suffix = {
            "composite": "comp",
            "lfe": "lfe",
            "nolfe": "nolfe",
        }[variant]
        material_filename = "diamond" if material_key == "Diamond" else material_key
        return (
            cls.qcdark2_root
            / "dielectric_functions"
            / variant
            / f"{material_filename}_{suffix}.h5"
        )

    # --- QCDark1 HDF5 form factors ---
    @classmethod
    def qcdark1_ff(cls, material: str) -> Path:
        return cls.form_factor_root / "QCDark" / f"{material}_final.hdf5"

    # --- QEDark txt form factors ---
    @classmethod
    def qedark_ff(cls, material: str) -> Path:
        return cls.form_factor_root / "QEDark" / f"{material}_f2.txt"

    # --- Noble gas pkl form factors ---
    @classmethod
    def noble_ff(cls, material: str) -> Path:
        return cls.form_factor_root / "wimprates" / \
               f"{material}_dme_ionization_ff.pkl"

    # --- Small bundled data ---
    p100K_dat: Path = _REPO_ROOT / "DMeRates" / "p100K.dat"

    # --- SRDM (Solar-Reflected Dark Matter) flux files ---
    @classmethod
    def srdm_manifest(cls) -> Path:
        """Path to halo_data/srdm/manifest.json."""
        return cls.halo_root / "srdm" / "manifest.json"

    @classmethod
    def srdm_flux_file(cls, filename: str) -> Path:
        """Path to a flux file under halo_data/srdm/."""
        return cls.halo_root / "srdm" / filename
