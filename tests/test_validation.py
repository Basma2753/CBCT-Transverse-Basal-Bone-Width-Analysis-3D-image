import numpy as np
import pytest
from cbct_width.validation import _lin_ccc, _icc_2_1


def test_agreement_is_one_for_identical_nonconstant_values():
    values = np.array([35., 42., 49., 51.])
    assert _lin_ccc(values, values) == pytest.approx(1)
    assert _icc_2_1(values, values) == pytest.approx(1)


def test_agreement_penalizes_fixed_bias():
    values = np.array([35., 42., 49., 51.])
    assert _lin_ccc(values, values + 10) < 0.8
    assert _icc_2_1(values, values + 10) < 0.8


def test_icc_handles_insufficient_or_constant_data():
    assert _icc_2_1([1, 2], [1, 2]) is None
    assert _icc_2_1([1, 1, 1], [1, 1, 1]) is None
