import numpy as np
import pytest

from burstqpo import bayes_factor


def test_bayes_factor_from_ordinary_evidences():
    assert bayes_factor(20.0, 5.0) == pytest.approx(4.0)


def test_bayes_factor_from_log_evidences():
    assert bayes_factor(np.log(20.0), np.log(5.0), log_input=True) == pytest.approx(4.0)


def test_bayes_factor_rejects_invalid_evidences():
    with pytest.raises(ValueError):
        bayes_factor(1.0, 0.0)
    with pytest.raises(ValueError):
        bayes_factor(np.nan, 1.0)
