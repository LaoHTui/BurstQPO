"""BurstQPO LSP/GLSP tutorial using a synthetic periodic signal."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burstqpo import LombScargleResult, lomb_scargle


def main() -> None:
    rng = np.random.default_rng(42)
    time = np.sort(rng.uniform(0.0, 20.0, 500))
    true_frequency = 1.5
    uncertainty = 0.45 + 0.2 * rng.random(len(time))
    values = 3.0 * np.sin(2.0 * np.pi * true_frequency * time)
    values += rng.normal(0.0, uncertainty)

    # auto + uncertainty selects GLSP. Omit uncertainty to select ordinary LSP.
    result = lomb_scargle(
        time, values, uncertainty=uncertainty, mode="auto",
        frequency_min=0.1, frequency_max=5.0, frequency_step=0.002,
        time_unit="d",
    )
    strongest = result.find_peaks(top_n=1)[0]
    print(f"Selected mode: {result.mode.upper()}")
    print(f"Strongest peak: {strongest['frequency']:.4f} Hz")

    output_dir = Path(__file__).resolve().parent
    result.plot_lomb_scargle(
        save_path=output_dir / "lomb_scargle_example.png",
        show_peaks=True,
    )
    result.save_npz(output_dir / "lomb_scargle_example.npz")
    restored = LombScargleResult.load_npz(output_dir / "lomb_scargle_example.npz")
    print(f"Restored mode: {restored.mode.upper()}")


if __name__ == "__main__":
    main()
