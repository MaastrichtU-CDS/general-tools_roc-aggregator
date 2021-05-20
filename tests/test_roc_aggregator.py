""" Test the ROC aggregators.
"""

import numpy as np
import pytest

from roc_aggregator import roc_curve, precision_recall_curve

TOTAL_COUNT = [4, 5]
NEGATIVE_COUNT = [2, 3]
INPUT = ([], [], [], NEGATIVE_COUNT, TOTAL_COUNT)

PARTIAL_CM = np.array([[1, 0], [1, 1], [4, 3], [4, 4]])
THRESHOLDS = [0.4, 0.3, 0.2, 0.1]

@pytest.fixture()
def mock_validate_input(mocker):
    """ So we control time """
    return mocker.patch('roc_aggregator.validations.validate_input', return_value=None)

@pytest.fixture()
def mock_partial_cm(mocker):
    """ So we control time """
    return mocker.patch('roc_aggregator.partial_cm', return_value=(PARTIAL_CM, THRESHOLDS))

def test_roc_curve(mock_validate_input, mock_partial_cm):
    """ Test the roc_curve function.
    """
    fpr, tpr, thresholds_stack = roc_curve(*INPUT)

    assert not mock_validate_input.called
    # assert mock_validate_input.assert_called_with()
    mock_partial_cm.assert_called_with(*INPUT)
    assert all(fpr == [0.2, 0.2, 0.8, 0.8])
    assert all(tpr == [0, 0.25, 0.75, 1])
    assert thresholds_stack == THRESHOLDS

def test_precision_recall_curve(mock_validate_input, mock_partial_cm):
    """ Test the precision_recall_curve function.
    """
    pre, recall, thresholds_stack = precision_recall_curve(*INPUT)

    assert not mock_validate_input.called
    # assert mock_validate_input.assert_called_with()
    mock_partial_cm.assert_called_with(*INPUT)
    assert all(pre == [0, 0.5, 3/7, 0.5])
    assert all(recall == [0, 0.25, 0.75, 1])
    assert thresholds_stack == THRESHOLDS
