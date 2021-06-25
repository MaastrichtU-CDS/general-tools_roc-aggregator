""" Test the ROC aggregators.
"""

import numpy as np
import pytest

from roc_aggregator import roc_curve, precision_recall_curve, partial_cm

TOTAL_COUNT = [8, 10]
NEGATIVE_COUNT = [2, 5]

TPR = [[0, 3/6, 1], [0, 0, 1/5, 3/5, 3/5, 1]]
FPR = [[0, 1/2, 1], [0, 2/5, 2/5, 2/5, 4/5, 5/5]]
THRESHOLDS = [[1.3, 0.3, 0.1], [1.4, 0.4, 0.35, 0.3, 0.2, 0.1]]

INPUT = (TPR, FPR, THRESHOLDS, NEGATIVE_COUNT, TOTAL_COUNT)

THRESHOLDS_STACKED = np.array([1.4, 1.3, 0.4, 0.35, 0.3, 0.2, 0.1])
PARTIAL_CM = np.array([[0, 0], [0, 0], [2, 0], [2, 1], [3, 6], [5, 6], [7, 11]])

@pytest.fixture()
def mock_validate_input(mocker):
    """ So we control time """
    return mocker.patch('roc_aggregator.validations.validate_input', return_value=None)

@pytest.fixture()
def mock_partial_cm(mocker):
    """ So we control time """
    return mocker.patch('roc_aggregator.partial_cm', return_value=(PARTIAL_CM, THRESHOLDS_STACKED))

def test_roc_curve(mock_validate_input, mock_partial_cm):
    """ Test the roc_curve function.
    """
    fpr, tpr, thresholds_stack = roc_curve(*INPUT)

    assert not mock_validate_input.called
    # assert mock_validate_input.assert_called_with()
    mock_partial_cm.assert_called_with(*INPUT)
    assert all(fpr == [0, 0, 2/7, 2/7, 3/7, 5/7, 1])
    assert all(tpr == [0, 0, 0, 1/11, 6/11, 6/11, 1])
    assert all(thresholds_stack == THRESHOLDS_STACKED)

def test_precision_recall_curve(mock_validate_input, mock_partial_cm):
    """ Test the precision_recall_curve function.
    """
    pre, recall, thresholds_stack = precision_recall_curve(*INPUT)

    assert not mock_validate_input.called
    # assert mock_validate_input.assert_called_with()
    mock_partial_cm.assert_called_with(*INPUT)
    assert all(pre == [1, 1, 0, 1/3, 6/9, 6/11, 11/18])
    assert all(recall == [0, 0, 0, 1/11, 6/11, 6/11, 1])
    assert all(thresholds_stack == THRESHOLDS_STACKED)

def test_partial_cm():
    """ Test the partial confusion matrix function.
    """
    cm_partial, thresholds_stack = partial_cm(
        FPR, TPR, THRESHOLDS, NEGATIVE_COUNT, TOTAL_COUNT)

    assert (cm_partial == PARTIAL_CM).all()
    assert all(thresholds_stack == THRESHOLDS_STACKED)
