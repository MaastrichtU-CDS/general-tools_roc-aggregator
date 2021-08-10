# roc-aggregator

Aggregates multiple Receiver Operating Characteristic (ROC) curves
obtained from different sources into one global ROC. Additionally, it’s
also possible to calculate the aggregated precision-recall (PR) curve.

## Motivation

The ROC and the AUC (Area Under the Curve) can be essential metrics when
evaluating a model. In situations where there is a parallelization of
the model development, such as federated learning, it becomes relevant
to obtain precise measures for these metrics. These approaches usually
produce partial results that require aggregation methods to get the
complete picture. The ROCaggregator appears in this context, allowing to
compute the precise ROC curve from the partial results.

## Usage

Install the package using one of the following options:

- pip3: `pip3 install roc-aggregator`
- this repository `pip3 install .`

### Example:

A complete example of the usage of the roc-aggregator can be found [here](./roc_aggregator/examples/example.py).

- Obtain the global ROC curve from different sources by providing the false positive rate (fpr), true positive rate (tpr), thresholds (thresh), the total number of negative samples, and the total number of samples from each source:

```python
from roc_aggregator import roc_curve

fpr_1 = [0, 0, 0, 0, 0.002, ...] # false positive rate values for each threshold
tpr_1 = [0, 0.004, 0.008, 0.012, 0.016, ...] # true positive rate values for each threshold
thresh_1 = [0.9994038, 0.9986345, 0.99847864, 0.99575908, 0.99567612, ...] # thresholds used
negative_count_1 = np.count_nonzero(y1 == 0) # count the number of negative labels
total_count_1 = len(y1) # total number of labels

...
# ROC
fpr, tpr, thresh_stack = roc_curve(
    [fpr_1, fpr_2, ...],
    [tpr_1, tpr_2, ...],
    [thresh_1, thresh_2, ...],
    [negative_count_1, negative_count_2, ...],
    [total_count_1, total_count_2, ...]
)

# Precision-Recall
pre, recall, thresh_stack = precision_recall_curve(
    [fpr_1, fpr_2, ...],
    [tpr_1, tpr_2, ...],
    [thresh_1, thresh_2, ...],
    [negative_count_1, negative_count_2, ...],
    [total_count_1, total_count_2, ...]
)
```

- Calculate the AUC using numpy:
  
```python
import numpy as np

np.trapz(tpr, fpr)
```

### Visualization

```python
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.plot(fpr, tpr, color=color, label=label, linestyle=linestyle)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()
```

## Testing

Unit tests are available at [`/roc-aggregator/tests`](./tests).
Install the dependencies required and run the tests using `pytest` or `python3 setup.py test`.
