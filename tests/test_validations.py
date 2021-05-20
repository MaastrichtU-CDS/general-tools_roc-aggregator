""" Test for the validation functions.
"""

import numpy as np
import pytest

from dROC import validations

MEASURES = [
    np.ones(3),
    np.ones(4)
]
COUNT = np.ones(2)

tests_validate_input = {
    'test_valid': (
        (MEASURES, MEASURES, MEASURES, COUNT, COUNT),
        None
    ),
    'test_invalid_fpr': (
        ([np.ones(3)], MEASURES, MEASURES, COUNT, COUNT),
        "The sizes of the different inputs don't match!"
    ),
    'test_invalid_tpr': (
        (MEASURES, [np.ones(3)], MEASURES, COUNT, COUNT),
        "The sizes of the different inputs don't match!"
    ),
    'test_invalid_thresholds': (
        (MEASURES, MEASURES, [np.ones(3)], COUNT, COUNT),
        "The sizes of the different inputs don't match!"
    ),
    'test_invalid_negative_count': (
        (MEASURES, MEASURES, MEASURES, np.ones(3), COUNT),
        "The sizes of the different inputs don't match!"
    ),
    'test_invalid_total_count': (
        (MEASURES, MEASURES, MEASURES, COUNT, np.ones(3)),
        "The sizes of the different inputs don't match!"
    ),
    'test_invalid_fpr_by_roc': (
        ([np.ones(3), np.ones(5)], MEASURES, MEASURES, COUNT, COUNT),
        "The number of measures for the fpr, tpr, and thresholds don't match!"
    ),
    'test_invalid_tpr_by_roc': (
        (MEASURES, [np.ones(3), np.ones(5)], MEASURES, COUNT, COUNT),
        "The number of measures for the fpr, tpr, and thresholds don't match!"
    ),
    'test_invalid_thresholds_by_roc': (
        (MEASURES, MEASURES, [np.ones(3), np.ones(5)], COUNT, COUNT),
        "The number of measures for the fpr, tpr, and thresholds don't match!"
    ),
}

@pytest.mark.parametrize("params,error", tests_validate_input.values(), ids=tests_validate_input.keys())
def test_validate_input(params, error):
    """ Multiple tests to evaluate the validate_input function"""
    if error:
        with pytest.raises(TypeError) as exception:
            validations.validate_input(*params)
        assert str(exception.value) == error
    else:
        validations.validate_input(*params)
