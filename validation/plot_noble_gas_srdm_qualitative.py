"""Qualitative Xe SRDM spectra plot for bundled Fig. 22-style flux files.

This helper is intentionally not a regression test. It produces a quick
paper-style charge-spectrum plot for visual inspection while noble-gas SRDM
remains validation-limited pending external digitized reference points.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numericalunits as nu

from DMeRates.DMeRate import DMeRate
from DMeRates.data.registry import DataRegistry


def _fig22_entries():
    with open(DataRegistry.srdm_manifest()) as f:
        entries = json.load(f)["files"]
    return [
        entry for entry in entries
        if "fig22" in entry.get("filename", "") and entry.get("FDMn") == 2
    ]


def main(output_path="tests/noble_gas_srdm_qualitative.png"):
    dm = DMeRate("Xe", form_factor_type="wimprates")
    ne_bins = list(range(1, 21))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for entry in sorted(_fig22_entries(), key=lambda item: item["mX_eV"]):
        rates = dm.calculate_rates(
            mX_array=[entry["mX_eV"] / 1.0e6],
            halo_model="srdm",
            FDMn=2,
            ne=ne_bins,
            sigma_e=entry["sigma_e_cm2"],
        )[:, 0]
        y = (rates * nu.kg * nu.year).detach().cpu().numpy()
        label = f"{entry['nominal_mX_eV'] / 1.0e3:g} keV"
        ax.step(ne_bins, y, where="mid", label=label)

    ax.set_yscale("log")
    ax.set_xlabel("electrons")
    ax.set_ylabel("events / kg-year")
    ax.set_title("Xe SRDM charge spectra, qualitative")
    ax.legend(title="nominal mχ")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    return output_path


if __name__ == "__main__":
    print(main())
