# Constants.py randomizes numericalunits unit scales at import time via
# random.uniform(). Seed Python's random module BEFORE DMeRates is imported so
# that every test session uses the same unit scales and reference values are
# reproducible. This must stay at module level — conftest.py is loaded before
# pytest imports test files.
import random
random.seed(0)

import pytest
import numpy as np


@pytest.fixture(scope='session', autouse=True)
def fix_units():
    """Documents the unit strategy for this test suite.

    Unit scales are fixed by random.seed(0) above, which runs before any
    DMeRates import in this session. Tests use nu.* values that were set by
    Constants.py with that seed, so conversions like `rates * nu.kg * nu.day`
    are consistent with the frozen reference values below.

    Do NOT call nu.reset_units() here — it would override Constants.py's
    already-computed derived constants and produce physically inconsistent
    conversions.
    """
    yield


# ---------------------------------------------------------------------------
# QEDark reference values
# Source: tests/qedark_validation.ipynb, cell 3
# Physics: form_factor_type='qedark', change_to_step(), integrate=False,
#          DoScreen=False, halo_model='imb', mX=[10, 1000] MeV
# Units:   rates.cpu().numpy() * nu.kg * nu.day  (seed-0 scales)
# Columns: [mX=10 MeV, mX=1000 MeV]; rows: ne=0..11 (Si) or ne=0..14 (Ge)
# ---------------------------------------------------------------------------
QEDARK_REFS = {
    ('Si', 0): np.array([
        [0.00000000e+00, 0.00000000e+00],
        [1.65988732e+03, 2.23609710e+01],
        [2.13363056e+03, 4.02343390e+01],
        [4.74983225e+02, 1.53541016e+01],
        [8.82161631e+01, 5.38279358e+00],
        [1.29268934e+01, 1.68072941e+00],
        [1.95046570e+00, 6.54079006e-01],
        [2.03960662e-01, 2.13696998e-01],
        [1.61351707e-02, 9.25843518e-02],
        [2.44331719e-04, 3.20550757e-02],
        [0.00000000e+00, 1.34429441e-02],
        [0.00000000e+00, 3.70147096e-03],
    ]),
    ('Ge', 2): np.array([
        [0.00000000e+00, 0.00000000e+00],
        [3.56263239e+02, 4.15840356e+00],
        [2.39502374e+02, 3.73339242e+00],
        [4.67344869e+01, 1.05436425e+00],
        [5.98025491e+00, 2.27092383e-01],
        [7.63103232e-01, 5.16267572e-02],
        [6.24593873e-02, 9.69667274e-03],
        [3.14249637e-03, 1.31711884e-03],
        [3.17456823e-04, 3.52119585e-04],
        [4.28697479e-04, 8.32054367e-04],
        [8.54170419e-05, 6.36781739e-04],
        [1.41085596e-06, 1.40766814e-04],
        [3.31868574e-09, 9.99530647e-05],
        [0.00000000e+00, 3.02901060e-05],
        [0.00000000e+00, 7.00461095e-06],
    ]),
}

# ---------------------------------------------------------------------------
# QCDark1 reference values
# Source: tests/qcdark_validation.ipynb, cell 3
# Physics: form_factor_type='qcdark', update_crosssection(1e-39), integrate=True,
#          DoScreen=True, halo_model='imb', mX=[1000] MeV
# Units:   rates.cpu().numpy() * nu.kg * nu.year  (seed-0 scales)
# Shape:   (ne, 1); rows: ne=0..10 (Si) or ne=0..14 (Ge)
# ---------------------------------------------------------------------------
QCDARK1_REFS = {
    ('Si', 0): np.array([
        [0.        ],
        [7.29164141],
        [5.78381139],
        [4.35673506],
        [2.53768945],
        [1.61129031],
        [1.14310557],
        [0.85278798],
        [0.65363051],
        [0.54000484],
        [0.42610991],
    ]),
    ('Ge', 2): np.array([
        [0.00000000e+00],
        [2.36651902e-01],
        [5.27697071e-01],
        [1.77457556e-01],
        [4.36148016e-02],
        [8.73402262e-03],
        [1.84050246e-03],
        [5.14887840e-04],
        [1.96940814e-04],
        [1.34992038e-04],
        [5.92040730e-04],
        [1.96934008e-03],
        [1.03915907e-03],
        [7.32433076e-04],
        [7.08800989e-04],
    ]),
}

# ---------------------------------------------------------------------------
# Wimprates (noble gas) reference values
# Source: tests/wimprates_validation.ipynb, cells 4-5
# Physics: form_factor_type='wimprates', material='Xe', update_crosssection(4e-44),
#          halo_model='shm', mX=1000 MeV, nes=range(1,17), returnShells=False
# Units:   rates[:,0].cpu().numpy() * nu.tonne * nu.year  (seed-0 scales)
# Shape:   (16,); index = ne-1, ne in 1..16
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# QCDark2 reference values
# Source: tests/qcdark2_formula_derivation.ipynb, sections 4 + 7
# Physics: material=Si (Si_comp.h5 composite dielectric), mX=1 GeV, mediator=heavy,
#          halo='MB', screening='RPA', default_astro (v0=238, vE=250.2, vesc=544 km/s,
#          rhoX=0.3 GeV/cm^3, sigma_e=1e-39 cm^2)
# Units:   events / kg / year / eV  (bare physical numbers; no nu seeding)
# ---------------------------------------------------------------------------
QCDARK2_REFS = {
    ('Si', 'heavy', 1e9): {
        # E [eV] : dR/dE [events/kg/year/eV]
          5.0: 1.84559612e+00,
         10.0: 1.00934857e+00,
         50.0: 5.29154460e-02,
    },
}


# ---------------------------------------------------------------------------
# QCDark2 SRDM reference values
# Source: tests/qcdark2_srdm_derivation.ipynb, Section 8
# Physics: material=Si (Si_comp.h5 composite dielectric), mX=48232.9466 eV
#          (nearest grid point to 50 keV), sigma_e=1.098541e-38 cm^2,
#          FDMn=2 (light mediator, m_A'=0), vector, screening=RPA
# Flux:   halo_data/srdm/srdm_dphidv_DPLM_row10_col8.txt
# Units:  events / kg / year / eV  (physical loader convention:
#         dPhi/d(v/c), integrated over dimensionless v/c)
# ---------------------------------------------------------------------------
QCDARK2_SRDM_REFS = {
    'Si_50keV_vector_light': [
        (  8.10, 2.549539e-01),
        ( 14.90, 6.807567e-01),
        ( 21.70, 5.155191e-01),
    ],
    'Si_50keV_scalar_light': [
        (  8.10, 2.546948868e-01),
        ( 14.90, 6.792294797e-01),
        ( 21.70, 5.146077259e-01),
    ],
    'Si_50keV_approx_light': [
        (  8.10, 2.546450944e-01),
        ( 14.90, 6.791123814e-01),
        ( 21.70, 5.144266838e-01),
    ],
    'Si_50keV_approx_full_light': [
        (  8.10, 2.549200157e-01),
        ( 14.90, 6.802473915e-01),
        ( 21.70, 5.153765292e-01),
    ],
}


WIMPRATES_REFS = {
    0: np.array([
        0.22374835, 0.15913665, 0.07393098, 0.09659583, 0.14614657,
        0.12005942, 0.07940309, 0.05126769, 0.03540802, 0.02520795,
        0.01670624, 0.01108186, 0.00750593, 0.00503709, 0.00340683,
        0.00227865,
    ]),
    2: np.array([
        7.83012459e-04, 5.37381554e-05, 5.52398315e-06, 1.06109174e-06,
        7.30696867e-07, 3.78628926e-07, 1.57482205e-07, 7.30807470e-08,
        3.88470105e-08, 2.28535837e-08, 1.23213702e-08, 6.66026395e-09,
        3.81525369e-09, 2.16962250e-09, 1.27168559e-09, 7.51045730e-10,
    ]),
}
