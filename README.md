# BurstQPO

BurstQPO 是一个面向伽马射线暴（Gamma-Ray Burst, GRB）及其他瞬变光变曲线的准周期振荡（Quasi-Periodic Oscillation, QPO）分析工具库。项目提供多种周期搜索方法、统一的结果对象、绘图与结果持久化，并包含基于高斯过程的贝叶斯建模和后验预测显著性估计。

> 当前版本为研究开发版。实际数据分析中应结合仪器曝光、时间窗、背景、死时间和搜索流程进行模拟标定；单个周期图峰值或受约束的 QPO 后验不能单独证明 QPO 探测。

想逐个了解函数、参数、返回值和完整分析流程，请阅读[详细使用指南](docs/USER_GUIDE.md)。

## 功能

| 方法 | 入口 | 适用数据 | 主要输出 |
| --- | --- | --- | --- |
| Weighted Wavelet Z-transform | `wwz` | 非均匀采样光变曲线 | 时频功率、COI、频率投影 |
| Lomb-Scargle / GLSP | `lomb_scargle` / `lsp` | 均匀或非均匀采样光变曲线 | 频率与功率谱 |
| Jurkevich | `jurkevich` | 光变曲线 | 周期与归一化方差 |
| Discrete Correlation Function | `dcf` | 光变曲线 | 时延、自相关与误差 |
| $Z_n^2$ | `z2n` | 未分箱事件到达时刻 | $Z_n^2$、局部与近似全局 p 值 |
| GP + MCMC | `run_mcmc` | 带测量误差的光变曲线 | 后验样本、诊断量、QPO 重建 |
| Nested sampling | `calculate_log_evidence` | H0/H1 贝叶斯模型 | log-evidence 与 Bayes factor |
| 后验预测显著性 | `calculate_significance` | 无 QPO 的 H0 后验 | 模拟标定的局部/全局显著性 |

所有周期搜索接口都返回结果对象。结果对象可用于寻找候选峰、绘图，并通过 `save_npz()` / `load_npz()` 保存和恢复。

## 安装

需要 Python 3.9 或更高版本。

```bash
git clone https://github.com/LaoHTui/BurstQPO.git
cd BurstQPO
python -m pip install -e .
```

WWZ 可选用 Numba 加速；开发环境可同时安装测试依赖：

```bash
python -m pip install -e ".[accelerated,dev]"
```

核心依赖包括 NumPy、SciPy、Matplotlib、Astropy、emcee 和 dynesty。

## 快速开始

下面的例子创建一个带噪声的非均匀采样周期信号，并用 GLSP 搜索主频：

```python
import numpy as np

from burstqpo import LombScargleResult, lomb_scargle

rng = np.random.default_rng(42)
time = np.sort(rng.uniform(0.0, 20.0, 500))
errors = 0.45 + 0.2 * rng.random(time.size)
values = 3.0 * np.sin(2.0 * np.pi * 1.5 * time)
values += rng.normal(0.0, errors)

result = lomb_scargle(
    time,
    values,
    uncertainty=errors,
    frequency_min=0.1,
    frequency_max=5.0,
    frequency_step=0.002,
    mode="auto",
    time_unit="s",
)

peak = result.find_peaks(top_n=1)[0]
print(f"frequency = {peak['frequency']:.4f} Hz")
print(f"period = {peak['period']:.4f} s")

result.plot_lomb_scargle(save_path="lomb_scargle.png", show_peaks=True)
result.save_npz("lomb_scargle.npz")
restored = LombScargleResult.load_npz("lomb_scargle.npz")
```

`mode="auto"` 在误差均为有限正数时选择 GLSP，否则选择普通 LSP。频率单位始终是输入时间单位的倒数。

## 方法示例

仓库中的示例均可从项目根目录直接运行：

| 示例 | 内容 |
| --- | --- |
| [`examples/lomb_scargle_tutorial.py`](examples/lomb_scargle_tutorial.py) | LSP/GLSP 周期搜索 |
| [`examples/wwz_tutorial.py`](examples/wwz_tutorial.py) | WWZ 时频分析与 COI |
| [`examples/jurkevich_tutorial.py`](examples/jurkevich_tutorial.py) | Jurkevich 周期搜索 |
| [`examples/dcf_tutorial.py`](examples/dcf_tutorial.py) | DCF 自相关搜索 |
| [`examples/z2n_tutorial.py`](examples/z2n_tutorial.py) | 事件序列的 $Z_n^2$ 搜索与解析显著性 |
| [`examples/gp_mcmc_tutorial.py`](examples/gp_mcmc_tutorial.py) | GP、MCMC、nested sampling 与 H0/H1 比较 |
| [`examples/lsp_significance_tutorial.py`](examples/lsp_significance_tutorial.py) | LSP 后验预测显著性标定 |

例如：

```bash
python examples/wwz_tutorial.py
python examples/z2n_tutorial.py
```

示例生成的 PNG、NPZ 和 JSON 结果默认不会提交到 Git。

## GP 与 MCMC

贝叶斯组件位于 `burstqpo.bayesian`。内置均值模型包括 FRED、ERCOD 以及常数、线性和多项式背景；协方差核包括红噪声核和指数阻尼余弦 QPO 核。

```python
from burstqpo.bayesian import (
    add_kernels,
    gp_log_likelihood,
    k_qpo,
    k_red_noise,
    set_prior,
)
from burstqpo.mean_model import add_mean_models, constant_background, fred

mean_function = add_mean_models(
    fred(parameter_indices=(0, 1, 2, 3, 4)),
    constant_background(parameter_index=5),
)
red_kernel = k_red_noise(parameter_indices=(6, 7))
qpo_kernel = k_qpo(parameter_indices=(8, 9, 10))
kernel_function = add_kernels(red_kernel, qpo_kernel)

model_group = {
    "mean_function": mean_function,
    "kernel_function": kernel_function,
}
```

`run_mcmc` 接收外部先验、似然和模型组件，返回 `MCMCResult`：

```python
from burstqpo import run_mcmc
from burstqpo.results import MCMCResult

result = run_mcmc(
    time,
    counts,
    errors,
    prior,
    {
        "initial": initial,
        "start_scale": start_scale,
        "n_walkers": 48,
        "n_steps": 6000,
        "burn_in": 1500,
        "parameter_names": parameter_names,
    },
    gp_log_likelihood,
    model_group=model_group,
    output_path="posterior.npz",
    seed=42,
)

print(result.mean_acceptance_fraction)
print(result.posterior_means)
print(result.posterior_1sigma)
print(result.maximum_log_likelihood, result.aic, result.bic)

result.plot_corner(save_path="posterior_corner.png")
restored = MCMCResult.load_npz("posterior.npz")
```

`MCMCResult` 还提供 `summary()`、最大后验/最大似然参数、`qpo_summary()` 和 `predict_qpo()`。其中 `maximum_log_likelihood` 是保留链样本中的最大似然值，并非额外的连续优化结果。

### Bayesian evidence

QPO 模型应与使用相同数据和均值模型的无 QPO 模型进行比较：

```python
from burstqpo.bayesian import bayes_factor, calculate_log_evidence

logz_h0, err_h0, nested_h0 = calculate_log_evidence(
    time,
    counts,
    errors,
    gp_log_likelihood,
    model_group_h0,
    prior_transform_h0,
    ndim=len(initial_h0),
)
logz_h1, err_h1, nested_h1 = calculate_log_evidence(
    time,
    counts,
    errors,
    gp_log_likelihood,
    model_group_h1,
    prior_transform_h1,
    ndim=len(initial_h1),
)
bf10 = bayes_factor(logz_h1, logz_h0, log_input=True)
```

不带联合约束的 `set_prior(...)` 会提供 `prior.prior_transform`。若存在 `log_c_qpo < log_f_qpo` 一类联合约束，应自定义与分析中归一化先验一致的 unit-cube transform。

## 后验预测显著性

显著性模拟必须来自不含 QPO 核的 H0 后验。新的 MCMC NPZ 文件包含观测数组，以及恢复内置均值模型和 GP 核所需的元数据。

```python
from burstqpo import lsp
from burstqpo.significance import (
    SignificanceResult,
    calculate_significance,
    simulate_light_curves,
)

simulations = simulate_light_curves(
    "fred_red_noise_h0.npz",
    n_simulations=1000,
    random_state=42,
    save_path="simulated_lc.npz",
)

significance = calculate_significance(
    "fred_red_noise_h0.npz",
    method=lsp,
    method_kwargs={
        "frequency_min": 10.0,
        "frequency_max": 40.0,
        "frequency_step": 0.01,
        "mode": "auto",
    },
    n_simulations=10000,
    random_state=42,
    save_path="lsp_significance.npz",
)

peak_index = int(significance.observed_statistic.argmax())
peak = significance.at(index=peak_index)
print(peak.local_pvalue, peak.global_pvalue)
print(peak.local_significance_sigma, peak.global_significance_sigma)

restored = SignificanceResult.load_npz("lsp_significance.npz")
```

内置 LSP/GLSP、WWZ、Jurkevich 和 DCF 可直接作为 `method`。自定义方法需要提供 `statistic_extractor(result) -> (coordinate, score)`，并保证更大的 `score` 表示更强的候选。$Z_n^2$ 使用未分箱事件时刻，不属于这一分箱光变曲线模拟接口。

## $Z_n^2$ 显著性

```python
from burstqpo import z2n

result = z2n(events, frequencies=frequencies, n_harmonics=2)
peak = result.find_peaks(top_n=1)[0]

print(peak["local_pvalue"])
print(peak["global_pvalue"])
print(result.n_trials)
print(result.z_local_threshold("0.1%"))
print(result.z_global_threshold("0.1%", n_trials=100))
```

固定频率下，均匀相位零假设给出渐近关系 $Z_n^2 \sim \chi^2(2n)$。默认全局 p 值使用 Sidak 校正，有效试验次数由搜索网格和频率过采样估计。复杂曝光窗、数据缺口、死时间、非平稳背景或小事件数会破坏简单近似，此时应使用保留完整观测与搜索流程的模拟标定。

## 绘图叠加曲线

五种周期搜索绘图方法均接受 `extra_curves`。每条曲线是一个包含 `values` 的字典，其余键会传给 Matplotlib：

```python
result.plot_lomb_scargle(
    extra_curves=[
        {
            "values": sigma_1,
            "label": "1 sigma",
            "color": "tab:orange",
            "linestyle": "--",
        },
        {"values": sigma_3, "label": "3 sigma", "color": "tab:red"},
    ]
)
```

标量 `values` 会自动扩展为阈值线。LSP、$Z_n^2$、Jurkevich 和 DCF 的数组长度应与图中横轴一致；WWZ 的数组长度应与 `result.frequency` 一致，并绘制在频率投影面板中。

## 项目结构

```text
burstqpo/
├── algorithms/       # 周期搜索算法
├── bayesian/         # GP、先验、MCMC 与 evidence
├── methods/          # 面向方法的兼容导入路径
├── models.py         # FRED、ERCOD 与背景模型
├── results.py        # 统一结果对象与 NPZ 持久化
├── significance.py   # 后验预测显著性
└── visualization.py  # 绘图接口
examples/             # 可直接运行的教学示例
tests/                # pytest 测试
```

## 开发与测试

```bash
python -m pip install -e ".[dev,accelerated]"
python -m pytest -q
```

当前测试覆盖周期搜索、绘图样式、结果持久化、MCMC 统计量、GP 模型、nested evidence 和后验预测显著性。
