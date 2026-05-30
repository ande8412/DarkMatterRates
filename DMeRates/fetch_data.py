"""Download QCDark2 dielectric HDF5 files from GitHub."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Optional, Sequence

_BASE_URL = (
    "https://raw.githubusercontent.com/meganhott/QCDark2/main/dielectric_functions"
)

_VARIANTS = ("composite", "lfe", "nolfe")

_FILES: dict[str, list[str]] = {
    "composite": [
        "GaAs_comp.h5",
        "Ge_comp.h5",
        "Si_comp.h5",
        "SiC_comp.h5",
        "diamond_comp.h5",
    ],
    "lfe": [
        "GaAs_lfe.h5",
        "Ge_lfe.h5",
        "Si_lfe.h5",
        "SiC_lfe.h5",
        "diamond_lfe.h5",
    ],
    "nolfe": [
        "GaAs_nolfe.h5",
        "Ge_nolfe.h5",
        "Si_nolfe.h5",
        "SiC_nolfe.h5",
        "diamond_nolfe.h5",
    ],
}

# Map lowercase material names to the filename stem used in QCDark2
_MATERIAL_ALIASES: dict[str, str] = {
    "si": "Si",
    "ge": "Ge",
    "gaas": "GaAs",
    "sic": "SiC",
    "diamond": "diamond",
}


def _stem_of(filename: str) -> str:
    """'Si_comp.h5' -> 'Si', 'diamond_comp.h5' -> 'diamond'"""
    return filename.split("_")[0]


def fetch_qcdark2(
    materials: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    dest: Optional[Path] = None,
    force: bool = False,
) -> list[Path]:
    """Download QCDark2 dielectric HDF5 files to *dest*.

    dest defaults to form_factors/QCDark2/ inside the DMeRates repo so the
    DataRegistry picks them up automatically without any env-var configuration.

    Args:
        materials: subset of {'Si', 'Ge', 'GaAs', 'SiC', 'Diamond'}.
                   Case-insensitive. Default: all five.
        variants:  subset of {'composite', 'lfe', 'nolfe'}.
                   Default: all three.
        dest:      root directory that will contain dielectric_functions/.
        force:     re-download files that already exist.

    Returns:
        List of paths to newly downloaded files.
    """
    from DMeRates.data.registry import DataRegistry

    if dest is None:
        dest = DataRegistry.bundled_qcdark2_root
    dest = Path(dest)

    variants_to_fetch = list(variants) if variants else list(_VARIANTS)
    invalid = set(variants_to_fetch) - set(_VARIANTS)
    if invalid:
        raise ValueError(
            f"Unknown variant(s): {sorted(invalid)}. Choose from {_VARIANTS}."
        )

    material_filter: Optional[set[str]] = None
    if materials:
        material_filter = set()
        for m in materials:
            key = m.lower()
            if key not in _MATERIAL_ALIASES:
                raise ValueError(
                    f"Unknown material '{m}'. "
                    f"Choose from: {sorted(_MATERIAL_ALIASES.keys())}."
                )
            material_filter.add(_MATERIAL_ALIASES[key])

    downloaded: list[Path] = []
    for variant in variants_to_fetch:
        for filename in _FILES[variant]:
            stem = _stem_of(filename)
            if material_filter is not None and stem not in material_filter:
                continue

            out_path = dest / "dielectric_functions" / variant / filename
            if out_path.exists() and not force:
                print(f"  skip  {variant}/{filename} (already present)")
                continue

            url = f"{_BASE_URL}/{variant}/{filename}"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  fetch {variant}/{filename} ...", end="", flush=True)
            try:
                _download(url, out_path)
                print(" done")
                downloaded.append(out_path)
            except Exception as exc:
                print(f" FAILED: {exc}")
                # Remove partial file so a retry doesn't see a corrupt stub.
                if out_path.exists():
                    out_path.unlink()
                raise

    return downloaded


def _download(url: str, dest: Path, chunk_size: int = 5 * 1024 * 1024) -> None:
    """Stream *url* to *dest*, printing a dot for every ~5 MB received."""
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        while True:
            data = response.read(chunk_size)
            if not data:
                break
            out.write(data)
            sys.stdout.write(".")
            sys.stdout.flush()
