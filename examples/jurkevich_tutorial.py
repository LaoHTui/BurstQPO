"""教学示例：使用 BurstQPO 运行 Jurkevich 周期搜索。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burstqpo import JurkevichResult, jurkevich


def main() -> None:
    rng = np.random.default_rng(11)
    time = np.sort(rng.uniform(0.0, 30.0, 700))
    true_period = 2.5
    values = np.sin(2.0 * np.pi * time / true_period)
    values += rng.normal(0.0, 0.18, len(time))
    periods = np.linspace(1.0, 5.0, 800)

    result = jurkevich(time, values, periods, m=10, time_unit="s")
    best = result.find_peaks(top_n=1)[0]
    print(f"Best period: {best['period']:.4f} +/- {best['period_err']:.4f} s")
    print(f"Normalized variance: {best['v_norm']:.5f}")

    output_dir = Path(__file__).resolve().parent
    result.plot_jurkevich(
        save_path=output_dir / "jurkevich_example.png",
        show_peaks=True,
    )
    result.save_npz(output_dir / "jurkevich_example.npz")
    restored = JurkevichResult.load_npz(output_dir / "jurkevich_example.npz")
    print(f"Restored phase bins: m={restored.m}")


if __name__ == "__main__":
    main()
