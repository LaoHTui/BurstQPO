"""教学示例：使用 BurstQPO 计算离散相关函数。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burstqpo import DCFResult, dcf


def main() -> None:
    rng = np.random.default_rng(23)
    time = np.sort(rng.uniform(0.0, 30.0, 500))
    period = 2.5
    values = np.sin(2.0 * np.pi * time / period)
    values += rng.normal(0.0, 0.2, len(time))

    result = dcf(time, values, delta_tau=0.05, c=0.12, max_tau=8.0, time_unit="s")
    peaks = result.find_peaks(top_n=2, min_period=0.5, distance=5)
    for peak in peaks:
        print(f"Lag: {peak['lag']:.4f} +/- {peak['lag_err']:.4f} s")

    output_dir = Path(__file__).resolve().parent
    result.plot_dcf(save_path=output_dir / "dcf_example.png", show_peaks=True)
    result.save_npz(output_dir / "dcf_example.npz")
    restored = DCFResult.load_npz(output_dir / "dcf_example.npz")
    print(f"Restored DCF bins: {len(restored.lag)}")


if __name__ == "__main__":
    main()
