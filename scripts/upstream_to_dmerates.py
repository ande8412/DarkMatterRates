#!/usr/bin/env python3
"""Convert upstream modulation/SRDM outputs to DMeRates loader layouts.

Usage
-----
Ordinary Halo-DM modulation, eta tables:
    python scripts/upstream_to_dmerates.py --product halo-modulated \\
        --source verne --FDMn 2 --mX-MeV 1.0 --sigma-e-cm2 1e-35 \\
        /path/to/verne/eta_run/ halo_data/

Direct SRDM, one angle-averaged flux:
    python scripts/upstream_to_dmerates.py --product srdm-direct \\
        --source damascus-sun --FDMn 2 --mX-MeV 1.0 --sigma-e-cm2 1e-35 \\
        /path/to/Differential_SRDM_Flux.txt halo_data/

Modulated SRDMBeam, Verne parameter directory:
    python scripts/upstream_to_dmerates.py --product srdmbeam-modulated \\
        --source verne --FDMn 2 \\
        /path/to/verne/results/FDMq2/SRDMBeam/mDM_1.0000MeV_sigmaE_1e-35cm2 \\
        halo_data/

Modulated SRDMBeam, DaMaSCUS flat run directory:
    python scripts/upstream_to_dmerates.py --product srdmbeam-modulated \\
        --source damascus --FDMn 2 \\
        /path/to/damascus/results/my_run/ \\
        halo_data/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from DMeRates.srdm.mediators import normalize_mediator_spin

_M_E_MEV = 0.51099895
_M_P_MEV = 938.27208816
_DAMASCUS_KM_NU = 5.067e18
_DAMASCUS_S_NU = 1.51905e24
_LOCAL_RHO_GEV_CM3 = 0.3


def _sigma_e_from_sigma_p(sigma_p_cm2: float, m_chi_MeV: float) -> float:
    """Derive sigma_e from sigma_p via reduced-mass ratio σ_e = σ_p·(μ_e/μ_p)²."""
    mu_e = m_chi_MeV * _M_E_MEV / (m_chi_MeV + _M_E_MEV)
    mu_p = m_chi_MeV * _M_P_MEV / (m_chi_MeV + _M_P_MEV)
    return sigma_p_cm2 * (mu_e / mu_p) ** 2


def _canonical_source(source: str) -> str:
    key = str(source).strip().lower().replace("-", "_")
    if key == "verne":
        return "Verne"
    if key in {"damascus", "damascus_srdmbeam"}:
        return "DaMaSCUS"
    if key in {"damascus_sun", "damascussun"}:
        return "DaMaSCUS-SUN"
    raise ValueError(
        f"Unsupported upstream source {source!r}; expected Verne, DaMaSCUS, "
        "or DaMaSCUS-SUN"
    )


def _fdm_dir(FDMn: int) -> str:
    if FDMn == 0:
        return "FDM1"
    if FDMn == 2:
        return "FDMq2"
    raise ValueError(f"Unsupported FDMn={FDMn}; expected 0 or 2")


def _format_mass(mX_MeV: float) -> str:
    return str(round(float(mX_MeV), 3)).replace(".", "_")


def _format_sigma(sigma_e_cm2: float) -> str:
    return str(float(format(float(sigma_e_cm2), ".3g")))


def _read_metadata_if_present(input_dir: Path) -> dict:
    meta_path = input_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def _float_from_metadata(raw: dict, *keys: str) -> float | None:
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return float(value)
    return None


def _parse_parameter_dir_tokens(input_dir: Path) -> tuple[float | None, float | None]:
    match = re.search(
        r"mDM_(?P<mass>.+?)_MeV_sigmaE_(?P<sigma>.+?)_cm2$",
        input_dir.name,
    )
    if match is None:
        return None, None
    mass = float(match.group("mass").replace("_", "."))
    sigma = float(match.group("sigma"))
    return mass, sigma


def _resolve_mx_sigma(
    input_path: Path,
    *,
    mX_MeV: float | None,
    sigma_e_cm2: float | None,
    raw_metadata: dict | None = None,
) -> tuple[float, float]:
    raw = raw_metadata or _read_metadata_if_present(input_path)
    parsed_mx, parsed_sigma = _parse_parameter_dir_tokens(input_path)
    resolved_mx = (
        mX_MeV
        if mX_MeV is not None
        else _float_from_metadata(raw, "mDM_MeV", "m_chi_MeV", "mX_MeV")
    )
    resolved_sigma = (
        sigma_e_cm2
        if sigma_e_cm2 is not None
        else _float_from_metadata(raw, "sigmaE_cm2", "sigma_e_cm2")
    )
    if resolved_mx is None:
        resolved_mx = parsed_mx
    if resolved_sigma is None:
        resolved_sigma = parsed_sigma
    if resolved_mx is None or resolved_sigma is None:
        raise ValueError(
            "Could not infer mX_MeV and sigma_e_cm2 from metadata or directory "
            "name; pass --mX-MeV and --sigma-e-cm2 explicitly."
        )
    return float(resolved_mx), float(resolved_sigma)


def _copy_indexed_files(
    indexed_paths: list[tuple[int, Path]],
    *,
    destination_path_for_index,
) -> None:
    for index, src in indexed_paths:
        dst = destination_path_for_index(index)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _discover_damascus_raw_eta(input_dir: Path) -> tuple[Path, Path]:
    rho_files = sorted(input_dir.glob("*.rho"))
    if len(rho_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one DaMaSCUS .rho file in {input_dir}; "
            f"found {len(rho_files)}"
        )
    simid = rho_files[0].stem
    histogram_dir = input_dir / f"{simid}_histograms"
    if not histogram_dir.is_dir():
        raise FileNotFoundError(
            f"No DaMaSCUS histogram directory found: {histogram_dir}"
        )
    return rho_files[0], histogram_dir


def _write_damascus_raw_eta_tables(
    input_dir: Path,
    out_dir: Path,
) -> int:
    """Convert raw DaMaSCUS eta/rho output using DaMaSCUS_helper.fix_eta rules."""
    rho_path, histogram_dir = _discover_damascus_raw_eta(input_dir)
    rho_data = np.loadtxt(rho_path, delimiter="\t")
    rho_data = np.atleast_2d(rho_data)
    if rho_data.shape[1] < 2:
        raise ValueError(f"DaMaSCUS rho file must have at least two columns: {rho_path}")
    rho_by_ring = np.asarray(rho_data[:, 1], dtype=float)

    eta_paths: list[tuple[int, Path]] = []
    for path in histogram_dir.glob("eta.*"):
        match = re.search(r"eta\.(\d+)$", path.name)
        if match:
            eta_paths.append((int(match.group(1)), path))
    if not eta_paths:
        raise FileNotFoundError(
            f"No raw DaMaSCUS eta files found in {histogram_dir}; "
            "expected eta.<ring_index>"
        )
    eta_paths.sort()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ring_index, eta_path in eta_paths:
        if ring_index >= len(rho_by_ring):
            raise ValueError(
                f"DaMaSCUS eta ring {ring_index} has no matching rho row in {rho_path}"
            )
        data = np.loadtxt(eta_path, delimiter="\t")
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError(
                f"DaMaSCUS eta file must have at least two columns: {eta_path}"
            )
        vmin_kms = data[:, 0] * _DAMASCUS_S_NU / _DAMASCUS_KM_NU
        density_scale = rho_by_ring[ring_index] / _LOCAL_RHO_GEV_CM3
        eta_s_per_km = data[:, 1] * _DAMASCUS_KM_NU / _DAMASCUS_S_NU
        eta_s_per_km *= density_scale
        if data.shape[1] > 3:
            eta_err_s_per_km = data[:, 3] * _DAMASCUS_KM_NU / _DAMASCUS_S_NU
            eta_err_s_per_km *= density_scale
        else:
            eta_err_s_per_km = np.zeros_like(eta_s_per_km)
        out = np.column_stack([vmin_kms, eta_s_per_km, eta_err_s_per_km])
        np.savetxt(out_dir / f"DM_Eta_theta_{ring_index}.txt", out, delimiter="\t")
    return len(eta_paths)


def _load_or_empty_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"files": []}


def _write_manifest_entry(manifest_path: Path, entry: dict) -> None:
    manifest = _load_or_empty_manifest(manifest_path)
    files = [
        existing for existing in manifest.get("files", [])
        if not (
            existing.get("filename") == entry["filename"]
            or (
                existing.get("mX_eV") == entry["mX_eV"]
                and existing.get("sigma_e_cm2") == entry["sigma_e_cm2"]
                and existing.get("FDMn") == entry["FDMn"]
                and existing.get("mediator_spin") == entry["mediator_spin"]
            )
        )
    ]
    files.append(entry)
    manifest["files"] = files
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def convert_halo_modulated(
    input_dir: Path,
    output_root: Path,
    FDMn: int,
    source: str,
    *,
    mX_MeV: float | None = None,
    sigma_e_cm2: float | None = None,
    summer: bool = False,
) -> Path:
    """Convert ordinary Verne/DaMaSCUS Halo-DM eta tables to DMeRates layout."""
    source_name = _canonical_source(source)
    if source_name not in {"Verne", "DaMaSCUS"}:
        raise ValueError("Halo-DM modulation source must be Verne or DaMaSCUS")
    if summer and source_name != "Verne":
        raise ValueError("summer=True is only supported for Verne ordinary modulation")

    raw = _read_metadata_if_present(input_dir)
    mX_MeV, sigma_e_cm2 = _resolve_mx_sigma(
        input_dir,
        mX_MeV=mX_MeV,
        sigma_e_cm2=sigma_e_cm2,
        raw_metadata=raw,
    )

    source_dir = f"{source_name}_summer" if summer else source_name
    out_dir = (
        output_root
        / "modulated"
        / _fdm_dir(FDMn)
        / source_dir
        / f"mDM_{_format_mass(mX_MeV)}_MeV_sigmaE_{_format_sigma(sigma_e_cm2)}_cm2"
    )
    index_paths: list[tuple[int, Path]] = []
    for p in input_dir.glob("DM_Eta_theta_*.txt"):
        match = re.search(r"_theta_(\d+)\.txt$", p.name)
        if match:
            index_paths.append((int(match.group(1)), p))
    index_paths.sort()

    converted_from_raw = False
    if index_paths:
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_indexed_files(
            index_paths,
            destination_path_for_index=lambda index: out_dir / f"DM_Eta_theta_{index}.txt",
        )
        ring_count = len(index_paths)
    elif source_name == "DaMaSCUS":
        ring_count = _write_damascus_raw_eta_tables(input_dir, out_dir)
        converted_from_raw = True
    else:
        raise FileNotFoundError(
            f"No Halo-DM eta files found in {input_dir}\n"
            "Expected files matching: DM_Eta_theta_*.txt"
        )

    if raw:
        out_meta = dict(raw)
        out_meta.update(
            {
                "product": "halo_modulated",
                "source": source_name,
                "m_chi_MeV": mX_MeV,
                "sigma_e_cm2": sigma_e_cm2,
                "FDMn": int(FDMn),
                "ring_count": ring_count,
                "converted_from_raw_damascus": converted_from_raw,
            }
        )
        (out_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    elif converted_from_raw:
        out_meta = {
            "product": "halo_modulated",
            "source": source_name,
            "m_chi_MeV": mX_MeV,
            "sigma_e_cm2": sigma_e_cm2,
            "FDMn": int(FDMn),
            "ring_count": ring_count,
            "converted_from_raw_damascus": True,
            "conversion_note": (
                "Applied DaMaSCUS_helper.fix_eta conversion: vmin *= s/km, "
                "eta *= km/s, eta_err *= km/s, and eta columns scaled by "
                "rho_ring / 0.3 GeV cm^-3."
            ),
        }
        (out_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    return out_dir


def convert_srdm_direct(
    input_file: Path,
    output_root: Path,
    FDMn: int,
    source: str,
    *,
    mX_MeV: float | None = None,
    mX_eV: float | None = None,
    sigma_e_cm2: float,
    mediator_spin: str = "vector",
    grid_family: str | None = None,
    flux_type: str = "isotropic_angle_averaged",
    filename: str | None = None,
) -> Path:
    """Convert one angle-averaged SRDM flux file and register it in manifest."""
    source_name = _canonical_source(source)
    if not input_file.is_file():
        raise FileNotFoundError(f"No SRDM flux file found: {input_file}")
    if mX_eV is None:
        if mX_MeV is None:
            raise ValueError("Direct SRDM import requires --mX-eV or --mX-MeV")
        mX_eV = float(mX_MeV) * 1.0e6
    else:
        mX_eV = float(mX_eV)
    mX_MeV = float(mX_eV) / 1.0e6
    mediator_spin = normalize_mediator_spin(mediator_spin)

    source_slug = source_name.lower().replace("-", "_")
    if filename is None:
        filename = (
            f"srdm_dphidv_{source_slug}_mchi_{_format_mass(mX_MeV)}_MeV_"
            f"sigmae_{_format_sigma(sigma_e_cm2)}.txt"
        )
    out_dir = output_root / "srdm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / filename
    shutil.copy2(input_file, out_file)

    manifest_path = out_dir / "manifest.json"
    entry = {
        "mX_eV": mX_eV,
        "sigma_e_cm2": float(sigma_e_cm2),
        "FDMn": int(FDMn),
        "mediator_spin": mediator_spin,
        "nominal_mX_eV": mX_eV,
        "nominal_sigma_e_cm2": float(sigma_e_cm2),
        "grid_index": None,
        "grid_family": grid_family or f"{source_slug}_direct",
        "filename": filename,
        "source": source_name,
        "upstream_filename": str(input_file),
        "retrieved": "local",
        "cross_section_convention": "sigma_e_bar",
        "flux_type": flux_type,
    }
    _write_manifest_entry(manifest_path, entry)
    return out_file


def convert_verne(input_dir: Path, output_root: Path, FDMn: int) -> Path:
    """Convert a Verne parameter directory to the DMeRates loader layout.

    Parameters
    ----------
    input_dir:
        Verne parameter directory containing metadata.json and isoangle flux files.
    output_root:
        DMeRates halo-data root (e.g. ``halo_data/``).
    FDMn:
        Mediator form-factor power (0 = heavy, 2 = light).

    Returns
    -------
    Path
        The output parameter directory under OUTPUT_ROOT.
    """
    from DMeRates.srdm.flux_loader import srdmbeam_flux_path, srdmbeam_parameter_dir

    meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {input_dir}")
    raw = json.loads(meta_path.read_text())

    mX_MeV = float(raw["mDM_MeV"])
    sigma_e_cm2 = float(raw["sigmaE_cm2"])

    # Discover ring indices from zero-padded Verne filenames.
    index_paths: list[tuple[int, Path]] = []
    for p in input_dir.glob("Differential_SRDM_Flux_*_isoangle_*.txt"):
        m = re.search(r"_isoangle_(\d+)\.txt$", p.name)
        if m:
            index_paths.append((int(m.group(1)), p))
    if not index_paths:
        raise FileNotFoundError(
            f"No Verne isoangle flux files found in {input_dir}\n"
            f"Expected files matching: Differential_SRDM_Flux_*_isoangle_*.txt"
        )
    index_paths.sort()

    out_dir = srdmbeam_parameter_dir(
        mX_MeV, sigma_e_cm2, FDMn, modulated_source="Verne", base_data_dir=output_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for ring_index, src in index_paths:
        dst = srdmbeam_flux_path(
            mX_MeV, sigma_e_cm2, FDMn, ring_index,
            modulated_source="Verne", base_data_dir=output_root,
        )
        shutil.copy2(src, dst)

    # Build output metadata: start from all upstream fields, then add/rename
    # the four keys the DMeRates loader contract requires.
    out_meta = dict(raw)
    out_meta["flux_type"] = "post_earth_detector"
    out_meta["ring_count"] = int(raw["num_angles"])
    out_meta["m_chi_MeV"] = mX_MeV
    out_meta["sigma_e_cm2"] = sigma_e_cm2
    out_meta["sigma_p_cm2"] = float(raw.get("sigmaP_cm2") or raw.get("sigma_p_cm2"))
    out_meta["detector_depth_m"] = float(raw["depth_m"])
    out_meta["site_label"] = raw.get("site_label") or raw.get("target") or "unknown"
    # Normalise angle_convention to a string (Verne emits a dict).
    if isinstance(out_meta.get("angle_convention"), dict):
        out_meta["angle_convention"] = "0=overhead, 90=horizon, 180=nadir"
    out_meta["angle_grid_type"] = "point"
    # file_isoangle_deg and gamma_internal_deg are already present in raw Verne metadata.

    (out_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    return out_dir


def convert_damascus(input_dir: Path, output_root: Path, FDMn: int) -> Path:
    """Convert a DaMaSCUS flat run directory to the DMeRates loader layout.

    Parameters
    ----------
    input_dir:
        DaMaSCUS run directory containing metadata.json and theta flux files.
    output_root:
        DMeRates halo-data root (e.g. ``halo_data/``).
    FDMn:
        Mediator form-factor power (0 = heavy, 2 = light).

    Returns
    -------
    Path
        The output parameter directory under OUTPUT_ROOT.
    """
    from DMeRates.srdm.flux_loader import srdmbeam_flux_path, srdmbeam_parameter_dir

    meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {input_dir}")
    raw = json.loads(meta_path.read_text())

    mX_MeV = float(raw["m_chi_MeV"])
    sigma_p_cm2 = float(raw["sigma_p_cm2"])
    sigma_e_cm2 = _sigma_e_from_sigma_p(sigma_p_cm2, mX_MeV)

    index_paths: list[tuple[int, Path]] = []
    for p in input_dir.glob("Differential_SRDM_Flux_theta_*.txt"):
        m = re.search(r"_theta_(\d+)\.txt$", p.name)
        if m:
            index_paths.append((int(m.group(1)), p))
    if not index_paths:
        raise FileNotFoundError(
            f"No DaMaSCUS theta flux files found in {input_dir}\n"
            f"Expected files matching: Differential_SRDM_Flux_theta_*.txt"
        )
    index_paths.sort()

    out_dir = srdmbeam_parameter_dir(
        mX_MeV, sigma_e_cm2, FDMn, modulated_source="DaMaSCUS", base_data_dir=output_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for ring_index, src in index_paths:
        dst = srdmbeam_flux_path(
            mX_MeV, sigma_e_cm2, FDMn, ring_index,
            modulated_source="DaMaSCUS", base_data_dir=output_root,
        )
        shutil.copy2(src, dst)

    # DaMaSCUS metadata keys already match DMeRates conventions; only patch the
    # two fields that are wrong or absent in raw DaMaSCUS output.
    out_meta = dict(raw)
    out_meta["sigma_e_cm2"] = sigma_e_cm2
    # angle_to_ring_mapping is present but angle_grid_type is absent; setting it
    # explicitly prevents the loader's floor-mapping heuristic from emitting a warning.
    out_meta["angle_grid_type"] = "bin_average"

    (out_dir / "metadata.json").write_text(json.dumps(out_meta, indent=2) + "\n")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert upstream Verne, DaMaSCUS, or DaMaSCUS-SUN output into "
            "DMeRates-native halo_data layouts."
        )
    )
    parser.add_argument(
        "--product",
        default="srdmbeam-modulated",
        choices=["halo-modulated", "srdm-direct", "srdmbeam-modulated"],
        help=(
            "Output product to import. Defaults to srdmbeam-modulated for "
            "backward compatibility with the original converter."
        ),
    )
    parser.add_argument(
        "--source", required=True,
        choices=["verne", "damascus", "damascus-sun"],
        help="Upstream source type.",
    )
    parser.add_argument(
        "--FDMn", required=True, type=int, choices=[0, 2],
        help="Mediator form-factor power (0 = heavy mediator, 2 = light mediator).",
    )
    parser.add_argument("--mX-MeV", type=float, default=None, help="DM mass in MeV.")
    parser.add_argument("--mX-eV", type=float, default=None, help="DM mass in eV.")
    parser.add_argument(
        "--sigma-e-cm2",
        type=float,
        default=None,
        help="DM-electron cross section in cm^2.",
    )
    parser.add_argument(
        "--mediator-spin",
        default="vector",
        help="Direct SRDM mediator spin label for manifest entries.",
    )
    parser.add_argument(
        "--grid-family",
        default=None,
        help="Optional direct-SRDM manifest grid_family.",
    )
    parser.add_argument(
        "--flux-type",
        default="isotropic_angle_averaged",
        help="Optional direct-SRDM manifest flux_type.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional output filename for direct SRDM flux imports.",
    )
    parser.add_argument(
        "--summer",
        action="store_true",
        help="Import ordinary Verne summer modulation under Verne_summer.",
    )
    parser.add_argument(
        "input_path", type=Path,
        help=(
            "Upstream run directory for modulated products, or one flux file "
            "for srdm-direct."
        ),
    )
    parser.add_argument(
        "output_root", type=Path,
        help="DMeRates halo-data root directory (e.g. halo_data/).",
    )
    args = parser.parse_args()

    if args.product == "halo-modulated":
        out = convert_halo_modulated(
            args.input_path,
            args.output_root,
            args.FDMn,
            args.source,
            mX_MeV=args.mX_MeV,
            sigma_e_cm2=args.sigma_e_cm2,
            summer=args.summer,
        )
    elif args.product == "srdm-direct":
        if args.sigma_e_cm2 is None:
            raise ValueError("srdm-direct import requires --sigma-e-cm2")
        out = convert_srdm_direct(
            args.input_path,
            args.output_root,
            args.FDMn,
            args.source,
            mX_MeV=args.mX_MeV,
            mX_eV=args.mX_eV,
            sigma_e_cm2=args.sigma_e_cm2,
            mediator_spin=args.mediator_spin,
            grid_family=args.grid_family,
            flux_type=args.flux_type,
            filename=args.filename,
        )
    elif args.product == "srdmbeam-modulated":
        if args.source == "verne":
            out = convert_verne(args.input_path, args.output_root, args.FDMn)
        elif args.source == "damascus":
            out = convert_damascus(args.input_path, args.output_root, args.FDMn)
        else:
            raise ValueError(
                "srdmbeam-modulated import supports --source verne or damascus"
            )

    print(f"Written to: {out}")


if __name__ == "__main__":
    main()
