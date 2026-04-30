# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

DMeRates is a Python library for vectorized calculation of dark matter (DM) electron scattering rates in Si, Ge, Xe, and Ar. It uses PyTorch for GPU-accelerated computation and was developed primarily to study daily modulation of DM signals due to Earth scattering. The associated paper is [arXiv:2507.00344](http://arxiv.org/abs/2507.00344).

## Environment Setup

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The code automatically detects and uses CUDA GPUs. MPS (Apple Silicon) is intentionally disabled due to float32 precision limitations — it can be re-enabled manually by setting `device='mps'` on the `DMeRate` constructor.

QCDark2 dielectric data is resolved at runtime via (in priority order):
1. `DMERATES_QCDARK2_ROOT` environment variable
2. Sibling checkout at `../QCDark2`
3. Optional bundled copy (if present)

## Running Examples

The primary entry point for understanding usage is `DMeRates_Examples.ipynb`. Launch Jupyter and open it:

```bash
jupyter notebook DMeRates_Examples.ipynb
```

For modulation study figures (the paper's results), use:

```bash
jupyter notebook modulation_study/modulation_figures.ipynb
```

To regenerate modulated rates (requires halo data from Dryad):

```bash
jupyter notebook modulation_study/modulation_rates_generating.ipynb
```

Derivation and validation notebooks live in `validation/` (e.g. `qcdark2_formula_derivation.ipynb`, `srdm_fig22_validation.ipynb`).

## Architecture

### Core Package: `DMeRates/`

- **`Constants.py`** — All physical constants and SHM halo parameters (`v0`, `vEarth`, `vEscape`, `rhoX`), material properties (band gaps, atomic weights, Thomas-Fermi screening params, noble gas binding energies). Uses `numericalunits` with randomized unit scales to catch unit errors at runtime. Edit this file to change default SHM parameters.

- **`DM_Halo.py`** — `DM_Halo_Distributions` class. Implements SHM (`etaSHM`), Tsallis (`etaTsa`), and Double Power Law (`etaDPL`) velocity distributions. The tensor version (`eta_MB_tensor`) is used in GPU computation paths. `generate_halo_files()` writes precomputed η(v_min) data to `halo_data/`.

- **`DMeRate.py`** — `DMeRate` class, the main calculation engine:
  - Constructor initializes device, loads form factors, precomputes ionization probabilities
  - `calculate_rates(mX_array, halo_model, FDMn, ne, ...)` — main public API, dispatches to semiconductor or noble gas paths
  - `vectorized_dRdE(...)` — computes dR/dE for semiconductors via QCDark/QEDark form factors (Si/Ge)
  - `calculate_qcdark2_ne_rates(mX, halo_model, ne, mediator_spin, ...)` — QCDark2 dielectric path
  - `noble_dRdE(...)` → `rate_dme_shell(...)` — per-shell rates for nobles (Xe/Ar)
  - `calculate_semiconductor_rates(...)` / `calculate_nobleGas_rates(...)` — mass-array loops
  - `generate_dat(...)` — writes pre-calculated rates to `DMeRates/Rates/*.dat`
  - `setup_halo_data(mX, FDMn, halo_model, isoangle=...)` — loads the right η(v_min) file; generates it if missing

- **`rate_calculator.py`** — thin dispatcher that routes `calculate_rates` calls to the right engine based on `halo_model` and material.

- **`spectrum.py`** — converts dR/dE to observed electron spectra using ionization probabilities.

### Subpackages

- **`engines/`** — Calculation engines, one per form-factor/data source:
  - `dielectric.py` — QCDark2 native dielectric engine. Reproduces QCDark2's `get_dR_dE()` without importing `qcdark2.*`; formula derived in `validation/qcdark2_formula_derivation.ipynb`. Uses bare-float QCDark2 conventions internally.
  - `form_factor.py` — QCDark/QEDark crystal form factor engine (Si/Ge legacy path).
  - `noble_gas.py` — Noble gas SRDM engine for Xe/Ar shell-by-shell rates.

- **`screening/`** — Screening corrections applied to form factors:
  - `lindhard.py` — Zero-temperature RPA Lindhard dielectric screening for QEDark/QCDark1.
  - `semiconductor.py` — Semiconductor Thomas-Fermi screening.
  - `thomas_fermi.py` — TF screening utilities.
  - `dielectric.py` — Shared screening interface.

- **`srdm/`** — Sub-relativistic dark matter halo infrastructure:
  - `flux_loader.py` — Loads precomputed SRDM dΦ/dv flux files from `halo_data/srdm/` using the manifest.
  - `kinematics.py` — SRDM kinematics: γ(v), v_min(q, E, mX), H-vector, q bounds.
  - `manifest.py` — Parses `halo_data/srdm/manifest.json`; matches flux files by mass/cross-section with tolerance.
  - `mediators.py` — Mediator-spin normalization and flux-routing policy for vector/scalar/approx modes.

- **`responses/`** — Data loaders for each form-factor format:
  - `dielectric.py` — QCDark2 HDF5 dielectric response loader (`dielectric_response` class).
  - `dielectric_materials.py` — Material metadata for QCDark2 targets.
  - `noble_gas.py` — wimprates `.pkl` loader for Xe/Ar.
  - `qcdark1.py` — QCDark HDF5 loader.
  - `qedark.py` — QEDark `.txt` loader.

- **`halo/`** — Halo provider abstractions:
  - `analytic.py` — Wraps `DM_Halo_Distributions` for the `'shm'`, `'tsa'`, `'dpl'`, `'imb'` models.
  - `file_loader.py` — Loads precomputed η(v_min) files for `'modulated'` / `'summer'` / `'srdm'`.
  - `independent.py` — Halo-independent analysis helpers.

- **`ionization/`** — Electron-hole pair probability models:
  - `rk_probabilities.py` — Ramanathan-Kurinsky interpolated probabilities (Si, 100K).
  - `step_function.py` — Step-function approximation (Ge, always; Si optional).

- **`data/`** — Internal data registry:
  - `registry.py` — Resolves paths to QCDark2 HDF5 files, SRDM flux files, and manifest. Respects `DMERATES_QCDARK2_ROOT` env var.

### Data Directories

- **`form_factors/`** — Crystal form factors: `QCDark/` (HDF5), `QEDark/` (txt), `wimprates/` (pkl)
- **`halo_data/`** — Precomputed η(v_min) files for SHM and analytic models; `modulated/` holds DaMaSCUS/Verne angle-dependent files; `srdm/` holds SRDM dΦ/dv flux files and `manifest.json`
- **`DMeRates/Rates/`** — Pre-calculated rate `.dat` files, named by physics parameters
- **`limits/`** — Experimental constraint data (CSV files) and `Constraints.py` for loading them
- **`sensitivity_projections/`** — Expected sensitivity CSVs for Darkside-20k and Oscura
- **`modulation_study/`** — Analysis notebooks, `Modulation.py` (plotting/analysis utilities), and `isoangle.py`
- **`halo_independent/`** — Halo-independent analysis results and mock data
- **`validation/`** — Derivation and validation notebooks (e.g. `qcdark2_formula_derivation.ipynb`, `srdm_fig22_validation.ipynb`); tracked and committed
- **`torchinterp1d/`** — External submodule for GPU-compatible 1D interpolation (used in `get_halo_data` and `RKProbabilities`)

### Key Design Patterns

**Units**: All quantities carry `numericalunits` units throughout. To express a value in a specific unit, divide by it (e.g., `value / nu.km` gives km). The randomized unit scales in `Constants.py` act as a runtime unit-correctness test. Exception: `engines/dielectric.py` operates internally in bare QCDark2 floats (q in α·mₑ units, energies in eV) and converts at the boundary.

**Halo model string keys**: `'shm'`, `'tsa'`, `'dpl'` trigger analytic computation (or file lookup); `'modulated'` and `'summer'` use DaMaSCUS/Verne files indexed by `isoangle` (integer 0–35, representing 0°–175° in 5° steps); `'imb'` uses the in-memory Maxwell-Boltzmann tensor path; `'srdm'` loads precomputed sub-relativistic DM flux files from `halo_data/srdm/` via the manifest.

**Mediator spin** (`mediator_spin` parameter): controls the SRDM interaction mode. Canonical values are `'vector'`, `'scalar'`, `'approx'`, `'approx_full'` (alias: `'approx full'`). Noble gas SRDM only supports `'vector'`. Normalized by `srdm.mediators.normalize_mediator_spin()`.

**FDM form factor**: `FDMn=0` → heavy mediator (FDM=1); `FDMn=2` → light mediator (FDM∝1/q²). The parameter is the power `n` in `(α·me·c/q)^n`.

**Electron-hole pair probabilities**: Silicon uses interpolated Ramanathan-Kurinsky probabilities from `p100k.dat` (100K data). Germanium always uses the step function approximation (`change_to_step()` is called automatically).

**Integration**: The `integrate=True` path uses `torchquad.Simpson` for numerical q-integration; `integrate=False` uses a Riemann sum over the precomputed q-grid. QEDark form factors always use `integrate=False`. The QCDark2 dielectric engine uses a masked trapezoid matching QCDark2's Python-slice q convention.

**No qcdark2.* imports in production**: The QCDark2 formula is re-implemented from the derivation notebook. `grep -r "import qcdark2" DMeRates/` must return nothing (only `engines/dielectric.py` mentions it in comments).
