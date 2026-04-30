# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

## Current API Conventions (QCDark2 + SRDM)

- Public mass convention: `DMeRate.calculate_rates(...)` takes `mX_array` in **MeV** for every halo model, including `halo_model='srdm'`.
- Internal SRDM convention: SRDM manifest/engine keys are in **eV** (not public API units), including `halo_data/srdm/manifest.json`.
- QEDark/QCDark1 semiconductor screening accepts `screening='none'`, `screening='thomas_fermi'`, or `screening='lindhard'`. Omitting `screening` preserves legacy `DoScreen=True/False` behavior.
- Analytic Lindhard screening applies `1 / |epsilon_L(E,q)|^2` with default `lindhard_eta_eV=0.1`; Si/Ge `omegaP` and `qTF` come from arXiv:2306.14944 Table I, with `vF=sqrt(3) omegaP/qTF`.
- Lindhard-screening references: Lindhard 1954; arXiv:2404.10066 for the `1/|epsilon|^2` direct-detection correction; arXiv:2306.14944 Eq. (11)/Table I and Cappellini et al., Phys. Rev. B 47, 9892 (1993), for the existing analytical-screening constants/convention.
- QCDark2 requires explicit screening in public calls: pass `screening='rpa'` or `screening='none'`.
- Canonical SRDM flux-file source is:
  `https://github.com/hlxuhep/Solar-Reflected-Dark-Matter-Flux`.
- Support/missing-scope tracking: consult `tests/current_status.md` when present, plus the repo's “Still Missing / Not Yet Validated” status documentation.

## Architecture

### Core Package: `DMeRates/`

- **`Constants.py`** — All physical constants and SHM halo parameters (`v0`, `vEarth`, `vEscape`, `rhoX`), material properties (band gaps, atomic weights, Thomas-Fermi screening params, noble gas binding energies). Uses `numericalunits` with randomized unit scales to catch unit errors at runtime. Edit this file to change default SHM parameters.

- **`DM_Halo.py`** — `DM_Halo_Distributions` class. Implements SHM (`etaSHM`), Tsallis (`etaTsa`), and Double Power Law (`etaDPL`) velocity distributions. The tensor version (`eta_MB_tensor`) is used in GPU computation paths. `generate_halo_files()` writes precomputed η(v_min) data to `halo_data/`.

- **`form_factor.py`** — Three form factor classes:
  - `form_factor`: Loads QCDark `.hdf5` files for Si/Ge (primary)
  - `form_factorQEDark`: Loads QEDark `.txt` files for Si/Ge (legacy)
  - `formFactorNoble`: Loads wimprates `.pkl` files for Xe/Ar

- **`DMeRate.py`** — `DMeRate` class, the main calculation engine:
  - Constructor initializes device, loads form factors, precomputes ionization probabilities
  - `calculate_rates(mX_array, halo_model, FDMn, ne, ...)` — main public API, dispatches to semiconductor or noble gas paths
  - `vectorized_dRdE(...)` — computes dR/dE for semiconductors (Si/Ge)
  - `noble_dRdE(...)` → `rate_dme_shell(...)` — per-shell rates for nobles (Xe/Ar)
  - `calculate_semiconductor_rates(...)` / `calculate_nobleGas_rates(...)` — mass-array loops
  - `generate_dat(...)` — writes pre-calculated rates to `DMeRates/Rates/*.dat`
  - `setup_halo_data(mX, FDMn, halo_model, isoangle=...)` — loads the right η(v_min) file; generates it if missing

### Data Directories

- **`form_factors/`** — Crystal form factors: `QCDark/` (HDF5), `QEDark/` (txt), `wimprates/` (pkl)
- **`halo_data/`** — Precomputed η(v_min) files for SHM and other analytic models; `modulated/` subdirectory holds DaMaSCUS/Verne angle-dependent files for the modulation study
- **`DMeRates/Rates/`** — Pre-calculated rate `.dat` files, named by physics parameters
- **`limits/`** — Experimental constraint data (CSV files) and `Constraints.py` for loading them
- **`sensitivity_projections/`** — Expected sensitivity CSVs for Darkside-20k and Oscura
- **`modulation_study/`** — Analysis notebooks, `Modulation.py` (plotting/analysis utilities), and `isoangle.py`
- **`halo_independent/`** — Halo-independent analysis results and mock data
- **`torchinterp1d/`** — External submodule for GPU-compatible 1D interpolation (used in `get_halo_data` and `RKProbabilities`)

### Key Design Patterns

**Units**: All quantities carry `numericalunits` units throughout. To express a value in a specific unit, divide by it (e.g., `value / nu.km` gives km). The randomized unit scales in `Constants.py` act as a runtime unit-correctness test.

**Halo model string keys**: `'shm'`, `'tsa'`, `'dpl'` trigger analytic computation (or file lookup); `'modulated'` and `'summer'` use DaMaSCUS/Verne files indexed by `isoangle` (integer 0–35, representing 0°–175° in 5° steps); `'imb'` uses the in-memory Maxwell-Boltzmann tensor path.

**FDM form factor**: `FDMn=0` → heavy mediator (FDM=1); `FDMn=2` → light mediator (FDM∝1/q²). The parameter is the power `n` in `(α·me·c/q)^n`.

**Electron-hole pair probabilities**: Silicon uses interpolated Ramanathan-Kurinsky probabilities from `p100k.dat` (100K data). Germanium always uses the step function approximation (`change_to_step()` is called automatically).

**Integration**: The `integrate=True` path uses `torchquad.Simpson` for numerical q-integration; `integrate=False` uses a Riemann sum over the precomputed q-grid. QEDark form factors always use `integrate=False`.

---

## Current Support and Validation Status

Last updated: 2026-04-30, after noble-gas SRDM smoke coverage. Full detail in
[`tests/current_status.md`](tests/current_status.md).

### Fully supported and validated

- Legacy QEDark halo paths (SHM, Tsallis, DPL)
- Legacy QCDark1 halo paths (SHM, Tsallis, DPL)
- Noble gas (Xe/Ar) halo paths
- QCDark2 halo path (Si; dielectric HDF5 required at runtime)
- SRDM vector path for QCDark2 Si — benchmarked vs `../QCDark2` reference at <0.11% rel diff
- SRDM scalar/approx/approx\_full for QCDark2 Si — same benchmark, all modes <0.13%

### Implemented but validation-limited

- SRDM paths for QCDark1/QEDark (all mediator modes): smoke-tested, no external cross-code calibration
- Noble gas / wimprates SRDM: vector/dark-photon flux-tail path smoke-tested for Xe/Ar; no external numeric reference
- QEDark/QCDark1 screened SRDM: Thomas-Fermi and analytic Lindhard screening available; no independent screened-SRDM reference
- Non-Si QCDark2 materials and SRDM beyond Si: smoke coverage only
- Fig. 22 visual validation: **not accepted** — discrepancy not yet understood

### Explicitly missing after this merge

- Anisotropic or direction-dependent SRDM flux
- Auto-download/cache manager for SRDM flux archives; large-grid memory chunking
- Digitized Fig. 22 numerical regression; full Figs. 23+ constraints/projections reproduction
- Validated pair-energy constants for GaAs, SiC, Diamond

### Data-dependent validation

- Modulation notebook: requires Dryad/DaMaSCUS/Verne halo data (not in repo)
- QCDark2 runtime: requires dielectric HDF5 form-factor files (not in repo)
- Fig. 22 notebook: requires three DPLM flux files committed to `halo_data/srdm/`
