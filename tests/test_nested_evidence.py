import sys
import types

import numpy as np
import pytest

from burstqpo.bayesian.evidence import calculate_log_evidence


def test_calculate_log_evidence_runs_nested_sampler(monkeypatch):
    class FakeNestedResult:
        logz = np.array([-12.0, -10.5])
        logzerr = np.array([0.8, 0.2])

    class FakeSampler:
        def __init__(self, loglike, prior_transform, ndim, **kwargs):
            assert ndim == 2
            assert np.allclose(prior_transform([0.25, 0.75]), [0.25, 1.5])
            assert np.isfinite(loglike([0.2, 0.3]))
            self.results = FakeNestedResult()

        def run_nested(self, dlogz, print_progress):
            assert dlogz == pytest.approx(0.05)
            assert print_progress is False

    fake_dynesty = types.SimpleNamespace(NestedSampler=FakeSampler)
    monkeypatch.setitem(sys.modules, "dynesty", fake_dynesty)
    logz, error, nested = calculate_log_evidence(
        [0.0, 1.0], [1.0, 2.0], [0.1, 0.1],
        lambda theta, time, counts, errors: -1.0,
        {},
        lambda unit: np.asarray(unit) * [1.0, 2.0],
        2,
        dlogz=0.05,
    )
    assert logz == pytest.approx(-10.5)
    assert error == pytest.approx(0.2)
    assert nested.logz[-1] == pytest.approx(-10.5)
