"""SRDM flux file loaders.

This module is the only place that converts km/s -> v/c and applies the
dPhi/dv unit normalisation. The returned tensors follow numericalunits conventions.
"""
import json
import warnings
from pathlib import Path

import torch
import numpy as np
import numericalunits as nu
from DMeRates.data.registry import DataRegistry
from DMeRates.srdm.manifest import find_entry
from DMeRates.srdm.mediators import flux_mediator_spin, normalize_mediator_spin


_SRDMBEAM_SOURCE = "SRDMBeam"
_SRDMBEAM_LEGACY_SOURCE = "SRDMBeam"
_SRDMBEAM_UPSTREAM_SOURCES = {
    "verne": "Verne",
    "damascus": "DaMaSCUS",
    "damascus_srdmbeam": "DaMaSCUS",
    "srdmbeam": _SRDMBEAM_LEGACY_SOURCE,
}
_SRDMBEAM_FLUX_TYPE = "post_earth_detector"
_SRDMBEAM_REQUIRED_PROVENANCE_KEYS = (
    "angle_convention",
    "m_chi_MeV",
    "sigma_e_cm2",
    "site_label",
    "detector_depth_m",
)
_SRDMBEAM_UPSTREAM_INTERACTION_KEYS = (
    "sigma_p_cm2",
    "sigma_chi_p_cm2",
    "upstream_sigma_cm2",
    "interaction_cm2",
)


def _load_flux_table(flux_path: Path, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.loadtxt(str(flux_path), comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"{label} flux file must contain at least two columns: {flux_path}")

    v_kms = data[:, 0]      # km/s
    dphi_raw = data[:, 1]   # cm^-2 s^-1 (km/s)^-1

    positive_velocity = v_kms > 0.0
    if not np.any(positive_velocity):
        raise ValueError(f"{label} flux file has no positive-velocity rows: {flux_path}")
    v_kms = v_kms[positive_velocity]
    dphi_raw = dphi_raw[positive_velocity]

    # Convert velocity km/s -> dimensionless v/c
    # c_kms_bare is the speed of light expressed in km/s as a pure float
    c_kms_bare = nu.c0 / (nu.km / nu.s)
    v_over_c_np = v_kms / c_kms_bare

    # Convert dPhi/dv [cm^-2 s^-1 (km/s)^-1] -> dPhi/d(v/c) [nu units].
    # dPhi/d(v/c) = dPhi/dv_kms * c_kms_bare. The km/s factor must be a
    # bare number here; multiplying by nu.c0 would leak randomized units into
    # the flux normalization.
    dphi_dv_nu = dphi_raw * c_kms_bare / (nu.cm**2 * nu.s)

    v_tensor = torch.tensor(v_over_c_np, dtype=torch.float64)
    dphi_tensor = torch.tensor(dphi_dv_nu, dtype=torch.float64)

    return v_tensor, dphi_tensor


def format_srdmbeam_mass(mX_MeV: float) -> str:
    """Return the SRDMBeam filename mass token for public MeV masses."""
    return str(np.round(float(mX_MeV), 3)).replace(".", "_")


def format_srdmbeam_sigma(sigma_e_cm2: float) -> str:
    """Return the SRDMBeam filename cross-section token."""
    return str(float(format(sigma_e_cm2, ".3g")))


def srdmbeam_fdm_directory(FDMn: int) -> str:
    """Return the SRDMBeam FDM directory name for a mediator power."""
    if FDMn == 0:
        return "FDM1"
    if FDMn == 2:
        return "FDMq2"
    raise ValueError(f"Unsupported SRDMBeam FDMn={FDMn}; expected 0 or 2")


def _srdmbeam_base_dir(base_data_dir: str | Path | None = None) -> Path:
    if base_data_dir is None:
        return DataRegistry.halo_root
    return Path(base_data_dir)


def normalize_srdmbeam_modulated_source(modulated_source: str | None) -> str:
    """Return the canonical SRDMBeam upstream source/dataset selector.

    ``"Verne"`` and ``"DaMaSCUS"`` select the separated layouts:
    ``modulated/{FDM}/{source}/SRDMBeam/...``. ``"SRDMBeam"`` remains a legacy
    alias for the original flat layout ``modulated/{FDM}/SRDMBeam/...``.
    """
    if modulated_source is None:
        return _SRDMBEAM_LEGACY_SOURCE
    key = str(modulated_source).strip().lower().replace("-", "_")
    try:
        return _SRDMBEAM_UPSTREAM_SOURCES[key]
    except KeyError as exc:
        valid = ("Verne", "DaMaSCUS", _SRDMBEAM_LEGACY_SOURCE)
        raise ValueError(
            f"Unsupported SRDMBeam modulated_source={modulated_source!r}; "
            f"expected one of {valid}"
        ) from exc


def _srdmbeam_source_dir(
    FDMn: int,
    *,
    modulated_source: str = _SRDMBEAM_SOURCE,
    base_data_dir: str | Path | None = None,
) -> Path:
    source = normalize_srdmbeam_modulated_source(modulated_source)
    root = _srdmbeam_base_dir(base_data_dir) / "modulated" / srdmbeam_fdm_directory(FDMn)
    if source == _SRDMBEAM_LEGACY_SOURCE:
        return root / _SRDMBEAM_LEGACY_SOURCE
    return root / source / _SRDMBEAM_SOURCE


def srdmbeam_parameter_dir(
    mX_MeV: float,
    sigma_e_cm2: float,
    FDMn: int,
    *,
    modulated_source: str = _SRDMBEAM_SOURCE,
    base_data_dir: str | Path | None = None,
) -> Path:
    """Return the SRDMBeam parameter directory under the halo-data root."""
    mass_token = format_srdmbeam_mass(mX_MeV)
    sigma_token = format_srdmbeam_sigma(sigma_e_cm2)
    return (
        _srdmbeam_source_dir(
            FDMn,
            modulated_source=modulated_source,
            base_data_dir=base_data_dir,
        )
        / f"mDM_{mass_token}_MeV_sigmaE_{sigma_token}_cm2"
    )


def srdmbeam_flux_path(
    mX_MeV: float,
    sigma_e_cm2: float,
    FDMn: int,
    ring_index: int,
    *,
    modulated_source: str = _SRDMBEAM_SOURCE,
    base_data_dir: str | Path | None = None,
) -> Path:
    """Return the SRDMBeam flux table path for an integer ring index."""
    if not isinstance(ring_index, (int, np.integer)):
        raise ValueError(f"SRDMBeam ring_index must be an integer, got {ring_index!r}")
    if int(ring_index) < 0:
        raise ValueError(f"SRDMBeam ring_index must be non-negative, got {ring_index}")

    mass_token = format_srdmbeam_mass(mX_MeV)
    sigma_token = format_srdmbeam_sigma(sigma_e_cm2)
    filename = (
        f"Differential_SRDM_Flux_mDM_{mass_token}_MeV_"
        f"sigmaE_{sigma_token}_cm2_isoangle_{int(ring_index)}.txt"
    )
    return srdmbeam_parameter_dir(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source=modulated_source,
        base_data_dir=base_data_dir,
    ) / filename


def available_srdmbeam_ring_indices(
    mX_MeV: float,
    sigma_e_cm2: float,
    FDMn: int,
    *,
    modulated_source: str = _SRDMBEAM_SOURCE,
    base_data_dir: str | Path | None = None,
) -> list[int]:
    """Return sorted ring indices with flux files in the parameter directory."""
    parameter_dir = srdmbeam_parameter_dir(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source=modulated_source,
        base_data_dir=base_data_dir,
    )
    mass_token = format_srdmbeam_mass(mX_MeV)
    sigma_token = format_srdmbeam_sigma(sigma_e_cm2)
    prefix = f"Differential_SRDM_Flux_mDM_{mass_token}_MeV_sigmaE_{sigma_token}_cm2_isoangle_"
    indices = []
    for path in parameter_dir.glob(f"{prefix}*.txt"):
        suffix = path.stem.removeprefix(prefix)
        try:
            indices.append(int(suffix))
        except ValueError:
            continue
    return sorted(set(indices))


def infer_srdmbeam_ring_count(
    mX_MeV: float,
    sigma_e_cm2: float,
    FDMn: int,
    *,
    modulated_source: str = _SRDMBEAM_SOURCE,
    base_data_dir: str | Path | None = None,
) -> int:
    """Infer fixture-style ring count by counting available SRDMBeam flux files.

    Returns the number of flux files found, not max(ring_index) + 1. If ring
    indices have gaps (e.g. files for rings 0, 1, 3 but not 2), this returns 3,
    which may differ from the true ring_count in metadata. Use metadata.json
    ring_count for production data.
    """
    return len(
        available_srdmbeam_ring_indices(
            mX_MeV,
            sigma_e_cm2,
            FDMn,
            modulated_source=modulated_source,
            base_data_dir=base_data_dir,
        )
    )


def _validate_srdmbeam_metadata(
    raw_metadata: dict,
    *,
    mX_MeV: float,
    sigma_e_cm2: float,
    metadata_path: Path,
) -> int:
    flux_type = raw_metadata.get("flux_type")
    if flux_type != _SRDMBEAM_FLUX_TYPE:
        raise ValueError(
            f"SRDMBeam metadata flux_type must be {_SRDMBEAM_FLUX_TYPE!r}; "
            f"got {flux_type!r} in {metadata_path}"
        )

    ring_count = raw_metadata.get("ring_count")
    if not isinstance(ring_count, int) or ring_count <= 0:
        raise ValueError(
            f"SRDMBeam metadata ring_count must be a positive integer; "
            f"got {ring_count!r} in {metadata_path}"
        )

    missing = [
        key for key in _SRDMBEAM_REQUIRED_PROVENANCE_KEYS
        if key not in raw_metadata
    ]
    if not any(key in raw_metadata for key in _SRDMBEAM_UPSTREAM_INTERACTION_KEYS):
        missing.append("sigma_p_cm2 or equivalent upstream interaction field")
    if missing:
        warnings.warn(
            f"SRDMBeam metadata is missing production provenance fields "
            f"{missing} in {metadata_path}",
            RuntimeWarning,
            stacklevel=3,
        )

    metadata_mass = raw_metadata.get("m_chi_MeV")
    if metadata_mass is not None and not np.isclose(
        float(metadata_mass), float(mX_MeV), rtol=1e-6, atol=5e-4
    ):
        warnings.warn(
            f"SRDMBeam metadata m_chi_MeV={metadata_mass} does not match "
            f"lookup mX_MeV={mX_MeV} in {metadata_path}",
            RuntimeWarning,
            stacklevel=3,
        )

    metadata_sigma = raw_metadata.get("sigma_e_cm2")
    if metadata_sigma is not None and not np.isclose(
        float(metadata_sigma), float(sigma_e_cm2), rtol=5e-3, atol=0.0
    ):
        warnings.warn(
            f"SRDMBeam metadata sigma_e_cm2={metadata_sigma} does not match "
            f"lookup sigma_e_cm2={sigma_e_cm2} in {metadata_path}",
            RuntimeWarning,
            stacklevel=3,
        )

    return ring_count


def _validate_angle_array(
    raw_metadata: dict,
    key: str,
    *,
    expected_size: int,
    metadata_path: Path,
    strictly_increasing: bool,
) -> list[float]:
    try:
        values = np.asarray(raw_metadata[key], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"SRDMBeam metadata {key} must be a numeric 1D array "
            f"in {metadata_path}"
        ) from exc

    if values.ndim != 1 or values.size != expected_size:
        raise ValueError(
            f"SRDMBeam metadata {key} must contain {expected_size} values; "
            f"got shape {values.shape} in {metadata_path}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"SRDMBeam metadata {key} contains non-finite values in "
            f"{metadata_path}"
        )
    if np.any((values < 0.0) | (values > 180.0)):
        raise ValueError(
            f"SRDMBeam metadata {key} values must lie in [0, 180] degrees "
            f"in {metadata_path}"
        )
    if values.size > 1 and strictly_increasing and not np.all(np.diff(values) > 0.0):
        raise ValueError(
            f"SRDMBeam metadata {key} must be strictly increasing in "
            f"{metadata_path}"
        )

    return [float(value) for value in values]


def _normalize_angle_grid_type(raw_metadata: dict) -> str | None:
    raw_kind = (
        raw_metadata.get("angle_grid_type")
        or raw_metadata.get("angle_sampling")
        or raw_metadata.get("angle_coordinate_type")
    )
    if raw_kind is None:
        return None

    normalized = str(raw_kind).strip().lower().replace("-", "_").replace(" ", "_")
    point_kinds = {
        "point",
        "point_sample",
        "point_sampled",
        "points",
        "node",
        "nodes",
        "interpolation_node",
        "interpolation_nodes",
    }
    bin_kinds = {
        "bin",
        "bin_average",
        "bin_averaged",
        "ring_average",
        "ring_averaged",
        "histogram_bin",
        "histogram_bins",
    }
    if normalized in point_kinds:
        return "point"
    if normalized in bin_kinds:
        return "bin_average"
    raise ValueError(
        f"Unsupported SRDMBeam angle_grid_type={raw_kind!r}; expected a "
        f"point-sampled or bin-average angle grid"
    )


def _uniform_angle_bin_edges(ring_count: int) -> list[float]:
    return [float(edge) for edge in np.linspace(0.0, 180.0, ring_count + 1)]


def _angle_bin_centers(angle_bin_edges_deg: list[float]) -> list[float]:
    edges = np.asarray(angle_bin_edges_deg, dtype=float)
    return [float(value) for value in 0.5 * (edges[:-1] + edges[1:])]


def _resolve_srdmbeam_angle_metadata(
    raw_metadata: dict,
    *,
    ring_count: int,
    metadata_path: Path,
) -> dict:
    """Resolve file index angles without assuming points and bins are the same."""
    angle_grid_type = _normalize_angle_grid_type(raw_metadata)
    has_file_angles = "file_isoangle_deg" in raw_metadata
    has_bin_edges = "angle_bin_edges_deg" in raw_metadata

    if angle_grid_type == "point" or (angle_grid_type is None and has_file_angles):
        file_angles = _validate_angle_array(
            raw_metadata,
            "file_isoangle_deg",
            expected_size=ring_count,
            metadata_path=metadata_path,
            strictly_increasing=True,
        )
        return {
            "angle_grid_type": "point",
            "file_isoangle_deg": file_angles,
            "angle_bin_edges_deg": None,
            "angle_bin_centers_deg": None,
            "angle_representative_deg": file_angles,
        }

    if angle_grid_type == "bin_average" or (angle_grid_type is None and has_bin_edges):
        if has_bin_edges:
            bin_edges = _validate_angle_array(
                raw_metadata,
                "angle_bin_edges_deg",
                expected_size=ring_count + 1,
                metadata_path=metadata_path,
                strictly_increasing=True,
            )
        else:
            bin_edges = _uniform_angle_bin_edges(ring_count)
        bin_centers = _angle_bin_centers(bin_edges)
        return {
            "angle_grid_type": "bin_average",
            "file_isoangle_deg": None,
            "angle_bin_edges_deg": bin_edges,
            "angle_bin_centers_deg": bin_centers,
            "angle_representative_deg": bin_centers,
        }

    mapping = str(raw_metadata.get("angle_to_ring_mapping", "")).lower()
    if "floor" in mapping and "ring_count" in mapping:
        bin_edges = _uniform_angle_bin_edges(ring_count)
        bin_centers = _angle_bin_centers(bin_edges)
        warnings.warn(
            f"SRDMBeam metadata has floor-style angle_to_ring_mapping but no "
            f"explicit angle_grid_type in {metadata_path}; inferring "
            f"bin-average angular data and using bin centers for interpolation.",
            RuntimeWarning,
            stacklevel=3,
        )
        return {
            "angle_grid_type": "bin_average",
            "file_isoangle_deg": None,
            "angle_bin_edges_deg": bin_edges,
            "angle_bin_centers_deg": bin_centers,
            "angle_representative_deg": bin_centers,
        }

    raise ValueError(
        f"SRDMBeam metadata must specify either point-sampled "
        f"file_isoangle_deg or bin-average angle metadata in {metadata_path}"
    )


def load_srdmbeam_metadata(
    mX_MeV: float,
    sigma_e_cm2: float,
    FDMn: int,
    *,
    modulated_source: str = _SRDMBEAM_SOURCE,
    base_data_dir: str | Path | None = None,
) -> dict:
    """Load SRDMBeam metadata, falling back to fixture-style file counting."""
    parameter_dir = srdmbeam_parameter_dir(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source=modulated_source,
        base_data_dir=base_data_dir,
    )
    metadata_path = parameter_dir / "metadata.json"
    raw_metadata = {}

    if metadata_path.exists():
        with open(metadata_path) as f:
            raw_metadata = json.load(f)
        ring_count = _validate_srdmbeam_metadata(
            raw_metadata,
            mX_MeV=mX_MeV,
            sigma_e_cm2=sigma_e_cm2,
            metadata_path=metadata_path,
        )
        angle_metadata = _resolve_srdmbeam_angle_metadata(
            raw_metadata,
            ring_count=ring_count,
            metadata_path=metadata_path,
        )
    else:
        ring_count = infer_srdmbeam_ring_count(
            mX_MeV,
            sigma_e_cm2,
            FDMn,
            modulated_source=modulated_source,
            base_data_dir=base_data_dir,
        )
        if ring_count <= 0:
            raise FileNotFoundError(
                f"No SRDMBeam metadata.json or flux files found for "
                f"(mX_MeV={mX_MeV}, sigma_e_cm2={sigma_e_cm2}, FDMn={FDMn}) "
                f"in {parameter_dir}"
            )
        warnings.warn(
            f"SRDMBeam metadata.json is missing; inferred ring_count={ring_count} "
            f"by counting fixture-style flux files in {parameter_dir}. "
            f"Production flux loading requires provenance metadata.",
            RuntimeWarning,
            stacklevel=2,
        )
        angle_metadata = {
            "angle_grid_type": None,
            "file_isoangle_deg": None,
            "angle_bin_edges_deg": None,
            "angle_bin_centers_deg": None,
            "angle_representative_deg": None,
        }

    metadata = {
        "halo_model": "srdm_modulated",
        "flux_source": _SRDMBEAM_SOURCE,
        "modulated_source": normalize_srdmbeam_modulated_source(modulated_source),
        "upstream_source": normalize_srdmbeam_modulated_source(modulated_source),
        "parameter_dir": str(parameter_dir),
        "mX_MeV": float(mX_MeV),
        "mX_eV": float(mX_MeV) * 1e6,
        "sigma_e_cm2": float(sigma_e_cm2),
        "FDMn": int(FDMn),
        "ring_count": int(ring_count),
        "raw_metadata": raw_metadata,
    }
    metadata.update(angle_metadata)
    for key in (
        "flux_type",
        "angle_convention",
        "angle_to_ring_mapping",
        "angle_grid_type",
        "angle_sampling",
        "angle_coordinate_type",
        "angle_bin_edges_deg",
        "angle_bin_centers_deg",
        "angle_representative_deg",
        "file_isoangle_deg",
        "gamma_internal_deg",
        "m_chi_MeV",
        "sigma_p_cm2",
        "site_label",
        "detector_depth_m",
        "input_flux_file",
        "input_total_flux_cm2_s",
        "n_eff_input_cm3",
        "form_factor",
        "generation_timestamp",
    ):
        if key not in metadata:
            metadata[key] = raw_metadata.get(key)
    return metadata


def load_srdmbeam_flux(
    mX_MeV: float,
    sigma_e_cm2: float,
    FDMn: int,
    ring_index: int,
    *,
    modulated_source: str = _SRDMBEAM_SOURCE,
    base_data_dir: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load an SRDMBeam post-Earth detector flux table.

    Parameters use the public DMeRates convention: mass is in MeV and
    sigma_e_cm2 is the detector cross section used in filenames. ``ring_index``
    is the integer zero-based flux-file suffix, not a degree-valued solar angle.

    Returns:
        v_over_c : torch.float64 tensor, dimensionless v/c.
        dphi_dv  : torch.float64 tensor, dPhi/d(v/c) in numericalunits.
        metadata : dict containing raw metadata plus lookup/path fields.
    """
    flux_path = srdmbeam_flux_path(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        ring_index,
        modulated_source=modulated_source,
        base_data_dir=base_data_dir,
    )
    if not flux_path.exists():
        raise FileNotFoundError(
            f"No SRDMBeam flux file found for "
            f"(mX_MeV={mX_MeV}, sigma_e_cm2={sigma_e_cm2}, "
            f"FDMn={FDMn}, ring_index={ring_index}, "
            f"modulated_source={modulated_source!r}). "
            f"Attempted path: {flux_path}"
        )

    metadata = load_srdmbeam_metadata(
        mX_MeV,
        sigma_e_cm2,
        FDMn,
        modulated_source=modulated_source,
        base_data_dir=base_data_dir,
    )
    metadata["ring_index"] = int(ring_index)
    metadata["flux_file"] = str(flux_path)

    if int(ring_index) >= metadata["ring_count"]:
        raise ValueError(
            f"SRDMBeam ring_index={ring_index} is outside metadata ring_count="
            f"{metadata['ring_count']} for {metadata['parameter_dir']}"
        )
    file_angles = metadata.get("file_isoangle_deg")
    if file_angles is not None:
        metadata["ring_isoangle_deg"] = float(file_angles[int(ring_index)])
    representative_angles = metadata.get("angle_representative_deg")
    if representative_angles is not None:
        metadata["ring_representative_angle_deg"] = float(
            representative_angles[int(ring_index)]
        )

    v_tensor, dphi_tensor = _load_flux_table(flux_path, "SRDMBeam")
    return v_tensor, dphi_tensor, metadata


def load_srdm_flux(
    mX_eV: float,
    sigma_e_cm2: float,
    FDMn: int,
    mediator_spin: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load SRDM flux for the given (mX, sigma_e, FDMn, mediator_spin) point.

    Returns:
        v_over_c : torch.Tensor shape (N_v,), dimensionless v/c. Non-positive
                   velocity rows are dropped because SRDM cross sections contain
                   a 1/v^2 prefactor.
        dphi_dv  : torch.Tensor shape (N_v,), dPhi/d(v/c) in numericalunits.
                   Divide by 1/(nu.cm**2 * nu.s) to get the dimensionless
                   integration weight for a trapezoid integral over d(v/c).

    Raises:
        ValueError: if mediator_spin is invalid.
        FileNotFoundError: if no manifest entry matches the lookup tuple.
            Message includes the full lookup tuple AND the manifest path.
    """
    mediator_spin = normalize_mediator_spin(mediator_spin)
    flux_spin = flux_mediator_spin(mediator_spin)
    entry = find_entry(mX_eV, sigma_e_cm2, FDMn, mediator_spin)
    if entry is None:
        manifest_path = DataRegistry.srdm_manifest()
        raise FileNotFoundError(
            f"No SRDM flux file registered for "
            f"(mX_eV={mX_eV}, sigma_e_cm2={sigma_e_cm2}, "
            f"FDMn={FDMn}, mediator_spin={mediator_spin!r}, "
            f"flux_mediator_spin={flux_spin!r}). "
            f"See manifest at: {manifest_path}"
        )

    flux_path = DataRegistry.srdm_flux_file(entry["filename"])
    return _load_flux_table(flux_path, "SRDM")


def resolve_srdm_flux_source(
    *,
    source=None,
    mX_MeV: float | None = None,
    mX_eV: float | None = None,
    sigma_e_cm2: float,
    FDMn: int,
    mediator_spin: str,
    ring_index: int | None = None,
    base_data_dir: str | Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Resolve direct SRDM or SRDMBeam flux using one loader boundary.

    ``source`` is ``None``/``"direct"``/``"srdm"`` for the existing direct SRDM
    manifest path and ``"SRDMBeam"`` for post-Earth modulated SRDMBeam files.
    Both paths return dPhi/d(v/c) tensors in the same numericalunits convention.
    """
    mediator_spin = normalize_mediator_spin(mediator_spin)
    flux_spin = flux_mediator_spin(mediator_spin)
    source_key = "direct" if source in (None, "direct", "srdm") else source

    if source_key == "direct":
        if mX_eV is None:
            if mX_MeV is None:
                raise ValueError("Direct SRDM flux resolution requires mX_eV or mX_MeV")
            mX_eV = float(mX_MeV) * 1.0e6
        v_over_c, dphi_dv = load_srdm_flux(mX_eV, sigma_e_cm2, FDMn, mediator_spin)
        entry = find_entry(mX_eV, sigma_e_cm2, FDMn, mediator_spin)
        flux_file = None if entry is None else DataRegistry.srdm_flux_file(entry["filename"])
        metadata = {
            "halo_model": "srdm",
            "flux_source": "direct",
            "flux_file": None if flux_file is None else str(flux_file),
            "mX_eV": float(mX_eV),
            "mX_MeV": float(mX_eV) / 1.0e6,
            "sigma_e_cm2": float(sigma_e_cm2),
            "FDMn": int(FDMn),
            "mediator_spin": mediator_spin,
            "flux_mediator_spin": flux_spin,
        }
        return v_over_c, dphi_dv, metadata

    try:
        srdmbeam_source = normalize_srdmbeam_modulated_source(source_key)
    except ValueError:
        srdmbeam_source = None

    if srdmbeam_source is not None:
        if mX_MeV is None:
            if mX_eV is None:
                raise ValueError("SRDMBeam flux resolution requires mX_MeV or mX_eV")
            mX_MeV = float(mX_eV) / 1.0e6
        if ring_index is None:
            raise ValueError("SRDMBeam flux resolution requires an explicit integer ring_index")
        v_over_c, dphi_dv, metadata = load_srdmbeam_flux(
            mX_MeV,
            sigma_e_cm2,
            FDMn,
            int(ring_index),
            modulated_source=srdmbeam_source,
            base_data_dir=base_data_dir,
        )
        metadata["mediator_spin"] = mediator_spin
        metadata["flux_mediator_spin"] = flux_spin
        return v_over_c, dphi_dv, metadata

    raise ValueError(
        f"Unsupported SRDM flux source {source!r}; expected None, 'direct', "
        f"'Verne', 'DaMaSCUS', or {_SRDMBEAM_SOURCE!r}"
    )
