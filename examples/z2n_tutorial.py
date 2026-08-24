"""教学示例：使用 BurstQPO 计算事件序列的 Z²_n 周期图。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burstqpo import Z2nResult, z2n


def main() -> None:
    rng = np.random.default_rng(31)
    true_frequency = 1.25
    events = np.sort(rng.uniform(0.0, 40.0, 1800))
    # Event-rate modulation: retain events preferentially near the signal phase.
    keep_probability = 0.25 + 0.65 * (0.5 + 0.5 * np.sin(2.0 * np.pi * true_frequency * events))
    events = events[rng.random(len(events)) < keep_probability]
    frequencies = np.linspace(0.3, 2.2, 1200)

    result = z2n(events, frequencies=frequencies, n_harmonics=1, time_unit="s")
    peak = result.find_peaks(top_n=1)[0]
    print(f"Peak frequency: {peak['frequency']:.5f} +/- {peak['frequency_err']:.5f} Hz")

    output_dir = Path(__file__).resolve().parent
    result.plot_z2n(save_path=output_dir / "z2n_example.png", show_peaks=True)
    result.save_npz(output_dir / "z2n_example.npz")
    restored = Z2nResult.load_npz(output_dir / "z2n_example.npz")
    print(f"Restored harmonics: n={restored.n_harmonics}")


if __name__ == "__main__":
    main()
