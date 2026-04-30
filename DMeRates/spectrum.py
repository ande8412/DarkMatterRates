from dataclasses import dataclass, field
from typing import Optional, Dict
import torch


@dataclass
class RateSpectrum:
    """Container for a differential rate spectrum before detector-response conversion.

    E and dR_dE carry numericalunits units throughout. To express in a
    specific unit, divide by it (e.g., dR_dE / (1/(nu.kg * nu.year * nu.eV))).

    Metadata keys are backend-dependent. SRDM-producing engines additionally set
    keys such as: `halo_model`, `mediator_spin`, `flux_file`, `sigma_e_cm2`,
    and `variant`.
    """
    E: torch.Tensor               # energy array, shape (N_E,)
    dR_dE: torch.Tensor           # differential rate, shape (N_E,)
    material: str
    backend: str                  # 'qcdark1', 'qedark', 'qcdark2', 'noble_gas'
    metadata: Dict = field(default_factory=dict)
    # Optional shell decomposition for noble gases
    shell_spectra: Optional[Dict[str, torch.Tensor]] = None
    shell_labels: Optional[list] = None
