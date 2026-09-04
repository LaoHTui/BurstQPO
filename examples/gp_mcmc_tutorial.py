"""完整教学示例：用 emcee 和 nested sampling 比较两个 GP 假设。

在 BurstQPO 项目根目录运行：

    python examples/gp_mcmc_tutorial.py

H0 为 FRED + 背景 + 红噪声，H1 在 H0 上增加 QPO 核。示例先用
``emcee`` 检查 H1 后验，再分别用 nested sampling 计算 H0/H1 的
evidence 和 Bayes factor。这里使用短链和较小的 ``nlive`` 只为让示例
快速运行，不用于判断统计收敛或宣称 QPO 检测。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# 让示例可以直接从源码目录运行。安装 burstqpo 后可以去掉这两行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from burstqpo.bayesian.evidence import bayes_factor, calculate_log_evidence
from burstqpo.bayesian.gp import add_kernels, k_qpo, k_red_noise
from burstqpo.bayesian.likelihood import gp_log_likelihood
from burstqpo.bayesian.prior import set_prior
from burstqpo.mcmc import MCMCResult, run_mcmc
from burstqpo.mean_model import add_mean_models, constant_background, fred


PARAMETER_NAMES = (
    "amplitude",
    "t_max",
    "sigma_rise",
    "sigma_decay",
    "nu",
    "background",
    "log_a_red",
    "log_c_red",
    "log_a_qpo",
    "log_c_qpo",
    "log_f_qpo",
)

H0_PARAMETER_NAMES = PARAMETER_NAMES[:8]


def _build_problem():
    """Build one deterministic synthetic data set and both GP hypotheses."""
    rng = np.random.default_rng(20260826)
    time = np.linspace(-0.4, 0.6, 20)
    errors = np.full(time.size, 0.12)

    mean_function = add_mean_models(
        fred(parameter_indices=(0, 1, 2, 3, 4)),
        constant_background(parameter_index=5),
    )
    red_kernel = k_red_noise(parameter_indices=(6, 7))
    qpo_kernel = k_qpo(parameter_indices=(8, 9, 10))
    kernel_h0 = red_kernel
    kernel_h1 = add_kernels(red_kernel, qpo_kernel)

    initial_h1 = np.array([
        3.0, 0.0, 0.12, 0.25, 2.0, 1.0,
        np.log(0.08), np.log(3.0),
        np.log(0.03), np.log(2.0), np.log(8.0),
    ])
    covariance_h1 = kernel_h1(time, initial_h1) + np.diag(errors**2)
    counts = rng.multivariate_normal(
        mean_function(time, initial_h1), covariance_h1
    )

    return {
        "time": time,
        "counts": counts,
        "errors": errors,
        "mean_function": mean_function,
        "red_kernel": red_kernel,
        "qpo_kernel": qpo_kernel,
        "kernel_h0": kernel_h0,
        "kernel_h1": kernel_h1,
        "initial_h1": initial_h1,
    }


def _prior_specs_h0():
    """Normalized H0 prior in H0_PARAMETER_NAMES order."""
    return [
        ("uniform", 0.1, 8.0),
        ("uniform", -0.2, 0.2),
        ("uniform", 0.03, 0.4),
        ("uniform", 0.05, 0.8),
        ("uniform", 0.5, 4.0),
        ("uniform", 0.1, 3.0),
        ("uniform", np.log(1e-4), np.log(2.0)),
        ("uniform", np.log(0.1), np.log(20.0)),
    ]


def _prior_specs_h1():
    """Normalized H1 prior in PARAMETER_NAMES order.

    The QPO ranges satisfy ``c_qpo < f_qpo`` for every prior draw, so no
    additional non-normalized rejection constraint is needed here.
    """
    return _prior_specs_h0() + [
        ("uniform", np.log(1e-4), np.log(2.0)),
        ("uniform", np.log(0.1), np.log(5.0)),
        ("uniform", np.log(8.0), np.log(20.0)),
    ]


def run_nested_hypothesis_comparison(output_dir=None):
    """Compute H0/H1 nested evidences and their Bayes factor.

    This is intentionally a separate function so users can reuse the exact
    H0/H1 comparison with their own data and model functions.
    """
    problem = _build_problem()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "gp_mcmc_output"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prior_h0 = set_prior(_prior_specs_h0())
    prior_h1 = set_prior(_prior_specs_h1())
    time = problem["time"]
    counts = problem["counts"]
    errors = problem["errors"]

    # H0: no QPO covariance component.
    logz_h0, logzerr_h0, nested_h0 = calculate_log_evidence(
        time, counts, errors, gp_log_likelihood,
        {"mean_function": problem["mean_function"],
         "kernel_function": problem["kernel_h0"]},
        prior_h0.prior_transform,
        ndim=len(H0_PARAMETER_NAMES),
        nlive=60,
        dlogz=0.5,
        seed=20260826,
    )

    # H1: same mean/red-noise prior and data, plus the QPO kernel.
    logz_h1, logzerr_h1, nested_h1 = calculate_log_evidence(
        time, counts, errors, gp_log_likelihood,
        {"mean_function": problem["mean_function"],
         "kernel_function": problem["kernel_h1"]},
        prior_h1.prior_transform,
        ndim=len(PARAMETER_NAMES),
        nlive=60,
        dlogz=0.5,
        seed=20260827,
    )

    log_bf10 = logz_h1 - logz_h0
    log_bf10_error = float(np.hypot(logzerr_h0, logzerr_h1))
    bf10 = bayes_factor(logz_h1, logz_h0, log_input=True)
    comparison = {
        "H0": {"logZ": logz_h0, "logZ_error": logzerr_h0},
        "H1": {"logZ": logz_h1, "logZ_error": logzerr_h1},
        "logBF10": log_bf10,
        "logBF10_error": log_bf10_error,
        "BF10": bf10,
        "n_live": 60,
        "dlogz": 0.5,
        "parameter_names_H0": list(H0_PARAMETER_NAMES),
        "parameter_names_H1": list(PARAMETER_NAMES),
    }
    comparison_path = output_dir / "fred_red_noise_vs_qpo_evidence.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print("Nested evidence comparison saved to:", comparison_path)
    print("logZ(H0) =", logz_h0, "+/-", logzerr_h0)
    print("logZ(H1) =", logz_h1, "+/-", logzerr_h1)
    print("logBF10  =", log_bf10, "+/-", log_bf10_error)
    print("BF10     =", bf10)
    return comparison, nested_h0, nested_h1


def run_tutorial(output_dir=None):
    """运行教学短链，保存并重新读取后验结果。

    Parameters
    ----------
    output_dir : path-like or None
        输出目录。默认使用 ``examples/gp_mcmc_output``。

    Returns
    -------
    tuple
        ``(result, restored, posterior_path)``。
    """
    problem = _build_problem()
    time = problem["time"]
    counts = problem["counts"]
    errors = problem["errors"]
    mean_function = problem["mean_function"]
    qpo_kernel = problem["qpo_kernel"]
    kernel_function = problem["kernel_h1"]
    initial = problem["initial_h1"]
    start_scale = np.array([
        0.05, 0.005, 0.005, 0.008, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.03, 0.02,
    ])

    prior = set_prior(_prior_specs_h1())

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "gp_mcmc_output"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    posterior_path = output_dir / "fred_red_noise_qpo_posterior.npz"
    corner_path = output_dir / "fred_red_noise_qpo_corner.png"
    metadata_json_path = output_dir / "fred_red_noise_qpo_metadata.json"
    metadata_txt_path = output_dir / "fred_red_noise_qpo_metadata.txt"
    qpo_path = output_dir / "fred_red_noise_qpo_reconstruction.npz"

    result = run_mcmc(
        time=time,
        counts=counts,
        errors=errors,
        prior=prior,
        mcmc_config={
            "initial": initial,
            "start_scale": start_scale,
            "parameter_names": PARAMETER_NAMES,
            "n_walkers": 24,
            "n_steps": 40,
            "burn_in": 10,
            "thin": 2,
        },
        likelihood_function=gp_log_likelihood,
        model_group={
            "mean_function": mean_function,
            "kernel_function": kernel_function,
        },
        output_path=posterior_path,
        seed=20260826,
        model_id="fred_red_noise_qpo",
        metadata={"purpose": "GP-MCMC tutorial short chain"},
    )

    restored = MCMCResult.load_npz(posterior_path)
    np.testing.assert_allclose(restored.posterior, result.posterior)
    np.testing.assert_allclose(restored.log_probability, result.log_probability)
    if restored.parameter_names != PARAMETER_NAMES:
        raise RuntimeError("保存并读取后，参数名称发生了变化")

    # 1. 后验统计量：均值、中心 68% 区间、MAP 和链中最大似然样本。
    posterior_means = restored.posterior_means
    posterior_1sigma = restored.posterior_1sigma
    map_parameters = restored.maximum_a_posteriori_parameters
    mle_parameters = restored.maximum_likelihood_parameters
    result_summary = restored.summary()
    qpo_summary = restored.qpo_summary()

    print("Posterior saved to:", posterior_path)
    print("Posterior means:", posterior_means)
    print("Posterior 1-sigma:", posterior_1sigma)
    print("MAP log posterior:", restored.maximum_log_posterior)
    print("MLE log likelihood:", restored.maximum_log_likelihood)
    print("MAP parameters:", map_parameters)
    print("MLE parameters:", mle_parameters)
    print("QPO summary:", qpo_summary)
    print("AIC/BIC:", restored.aic, restored.bic)

    # 2. 保存可读元数据，并绘制所有参数的后验角图。
    restored.save_metadata_json(metadata_json_path)
    restored.save_metadata_txt(metadata_txt_path)
    restored.plot_corner(save_path=corner_path)

    # 3. 从每个（抽取的）后验参数条件化潜在 QPO GP 分量。
    #    mean 是 QPO 后验均值，std 是包含参数不确定性的总标准差。
    qpo_curve = restored.predict_qpo(
        time=time,
        counts=counts,
        errors=errors,
        mean_function=mean_function,
        kernel_function=kernel_function,
        qpo_kernel=qpo_kernel,
        prediction_time=np.linspace(time.min(), time.max(), 200),
        max_samples=64,
        random_state=20260826,
    )
    np.savez_compressed(
        qpo_path,
        time=qpo_curve["time"],
        mean=qpo_curve["mean"],
        std=qpo_curve["std"],
        covariance=qpo_curve["covariance"],
    )
    print("Corner plot saved to:", corner_path)
    print("Metadata JSON saved to:", metadata_json_path)
    print("Metadata TXT saved to:", metadata_txt_path)
    print("Latent QPO reconstruction saved to:", qpo_path)
    return result, restored, posterior_path


def main():
    run_tutorial()
    run_nested_hypothesis_comparison()


if __name__ == "__main__":
    main()
