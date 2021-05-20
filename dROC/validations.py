""" Functions to validate the inputs for the roc aggregator.
"""

def validate_input(fpr, tpr, thresholds, negative_count, total_count):
    """ Validate the input for the roc_curve function.
    """
    if not (len(fpr) == len(tpr) == len(thresholds) == len(negative_count) == len(total_count)):
        raise TypeError("The sizes of the different inputs don't match!")
    
    if not all([len(fpr[i]) == len(tpr[i]) == len(thresholds[i]) for i in range(len(fpr))]):
        raise TypeError(
            "The number of measures for the fpr, tpr, and thresholds don't match!")
