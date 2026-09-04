"""教学示例：使用 BurstQPO 检测合成光变曲线中的 QPO。

在 BurstQPO 项目根目录运行：

    python examples/wwz_tutorial.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# This keeps the example runnable directly from a source checkout. Installed
# users can omit it and run the same code as a normal Python module.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burstqpo.results import WWZResult
from burstqpo import wwz


def main() -> None:
    rng = np.random.default_rng(7)
    time = np.arange(0.0, 20.0, 0.02)
    qpo_frequency = 2.5

    # 频率从2.0 线性降到0.5，和time等长
    f_t = np.linspace(2, 0.5, len(time))

    trend = 30.0 + 0.4 * time
    # 瞬时相位：积分 f(t) dt，不能直接 f_t * time！
    dt = time[1] - time[0]
    phase = 2 * np.pi * qpo_frequency * np.cumsum(f_t) * dt
    signal = 10.0 * np.sin(phase)
    uncertainty = np.sqrt(trend) / 2.0
    count_rate = trend + signal + rng.normal(0.0, uncertainty)
    coefficients = np.polynomial.Polynomial.fit(time, count_rate, 1)
    values = count_rate - coefficients(time)
    frequencies = np.linspace(0.5, 5.0, 300)
    tau = np.linspace(-0.05, 0.3, 300)
    wwz_result = wwz(time, values, frequencies=frequencies, uncertainty=uncertainty,
                     tau = tau, c=0.0125)
    peaks = wwz_result.find_peaks(top_n=3, use_coi=True)

    print("候选 QPO:")
    for peak in peaks:
        print(f"  f={peak['frequency']:.3f} Hz, P={peak['period']:.3f} s, Z={peak['power']:.3f}")

    wwz_result.plot_wwz(save_path="wwz_example.png", show_coi=True,
                    projection_use_coi=True, show_projection=True, show_light_curve=True,
                    coi_alpha=0.16, show_peaks=True)
    wwz_result.save_npz("wwz_example.npz")
    # wwz_result.save_txt("wwz_example.txt")
    restored = WWZResult.load_npz("wwz_example.npz")
    print(restored.tau_number)
    print(f"WWZ window c = {restored.c}")
    # restored.plot_wwz(save_path="wwz_example_from_npz.png", show_coi=False)

    # WWZResult.load_npz("wwz_example.npz").plot_wwz(save_path="wwz_example_from_npz2.png")

    print("WWZ 图片已保存到 wwz_example.png")


if __name__ == "__main__":
    main()
