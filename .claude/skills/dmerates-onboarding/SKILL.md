---
name: dmerates-onboarding
description: Use when a physicist asks how to run dark matter rate calculations with DMeRates — setting up a calculation, understanding parameters (mX, FDMn, halo model, ne, mediator_spin), choosing a material or halo model, interpreting results, or finding the right notebook to start from.
---

# DMeRates Physics Guide

DMeRates calculates dark matter–electron scattering rates in Si, Ge, Xe, and Ar using PyTorch. The primary entry point is `DMeRates_Examples.ipynb` — always point users there first. It contains fully worked examples covering the standard workflow.

## Context to load

- `DMeRates_Examples.ipynb` — worked examples, the most reliable reference for how to set up and run a calculation
- `README.md` — environment setup and data source notes
- `CLAUDE.md` — full architecture and subpackage map, only if internal wiring details are needed

## Key physics parameters

**Material**: `'Si'`, `'Ge'`, `'Xe'`, `'Ar'`

**Dark matter mass** (`mX`): always in **MeV** at the public API level. Do not expose the internal eV convention to the user.

**Mediator** (`FDMn`): controls the momentum dependence of the DM form factor.
- `FDMn=0` — heavy mediator, F_DM = 1 (constant)
- `FDMn=2` — light mediator, F_DM ∝ 1/q²

**Halo model** (`halo_model`):
- `'shm'` — Standard Halo Model (default, most common starting point)
- `'tsa'` — Tsallis model
- `'dpl'` — Double Power Law
- `'srdm'` — Sub-relativistic (boosted) dark matter; requires precomputed flux files in `halo_data/srdm/` and a `sigma_e` argument
- `'modulated'` / `'summer'` — daily modulation study; requires DaMaSCUS/Verne files and an `isoangle` argument

**`ne`**: minimum number of electron–hole pairs detected. `ne=1` is the single-electron threshold; increase to model higher thresholds.

**`mediator_spin`** (SRDM only): interaction type for sub-relativistic DM.
- `'vector'` (default), `'scalar'`, `'approx'`, `'approx_full'`
- Noble gas SRDM (Xe/Ar) only supports `'vector'`

**Screening** (QCDark2 dielectric path for Si/Ge): `screening='rpa'` (physical, default) or `screening='none'` (unscreened, useful for comparison with older results).

## Common workflows

**Standard semiconductor rate** (Si or Ge, SHM halo):
```python
from DMeRates.DMeRate import DMeRate
import numpy as np

dm = DMeRate('Si')
mX_array = np.array([10., 20., 50., 100.])  # MeV
rates = dm.calculate_rates(mX_array, halo_model='shm', FDMn=0, ne=1)
```

**Noble gas rate** (Xe or Ar):
```python
dm = DMeRate('Xe')
rates = dm.calculate_rates(mX_array, halo_model='shm', FDMn=0, ne=1)
```

**Sub-relativistic DM (SRDM)**:
```python
dm = DMeRate('Si')
rates = dm.calculate_rates(
    mX_array, halo_model='srdm', FDMn=0, ne=1,
    mediator_spin='vector', sigma_e=1e-35  # sigma_e in cm²
)
```

## Custom halo velocity distribution

The constructor always loads the default SHM parameters from `Constants.py` (v₀ = 238 km/s, v_Earth = 250.2 km/s, v_esc = 544 km/s, ρ_X = 0.3 GeV/cm³). To override them, replace `dm.DM_Halo` after construction:

```python
import numericalunits as nu
from DMeRates.DMeRate import DMeRate
from DMeRates.DM_Halo import DM_Halo_Distributions

dm = DMeRate('Si')
dm.DM_Halo = DM_Halo_Distributions(
    V0    = 220.0 * nu.km / nu.s,
    VEarth= 244.0 * nu.km / nu.s,
    VEscape=533.0 * nu.km / nu.s,
    RHOX  = 0.3 * nu.GeV / nu.c0**2 / nu.cm**3,
)
rates = dm.calculate_rates(mX_array, halo_model='shm', FDMn=0, ne=1)
```

All four parameters must be given as `numericalunits` quantities (multiply the number by the unit). Any parameter left as `None` falls back to the Constants.py default.

## Output units (numericalunits)

`calculate_rates` returns an array whose values are in `numericalunits` internal representation — **not plain SI or CGS numbers**. The physical unit of the output is events / (kg · year).

To extract a plain numerical value, divide by the unit you want:

```python
import numericalunits as nu

rates = dm.calculate_rates(mX_array, halo_model='shm', FDMn=0, ne=1)

# events per kg per year (most common)
rates_per_kg_yr = rates / (1 / (nu.kg * nu.year))   # or: rates * nu.kg * nu.year

# events per gram per day
rates_per_g_day = rates / (1 / (nu.g * nu.day))

# events per tonne per year
rates_per_tonne_yr = rates / (1 / (1000 * nu.kg * nu.year))
```

The general rule in `numericalunits`: to express a quantity in unit U, divide by U. To attach a unit, multiply by U. The randomised unit scales in `Constants.py` act as a runtime check — if you forget to divide by a unit somewhere the answer will be random each run.

## Environment

```bash
source .venv/bin/activate
jupyter notebook DMeRates_Examples.ipynb
```

GPU (CUDA) is used automatically if available. On Apple Silicon, CPU is used by default (MPS disabled due to float32 precision limits).

## Common pitfalls

- Masses are **MeV** — passing eV values will produce rates many orders of magnitude off.
- SRDM requires both `sigma_e` (cm²) and the relevant flux files to be present in `halo_data/srdm/`. If files are missing, the manifest lookup will raise a clear error.
- `ne=1` is the lowest physical threshold; `ne=0` is not meaningful.
- QCDark2 screening must be specified explicitly when using the dielectric path: always pass `screening='rpa'` or `screening='none'`.
