# roc-aggregator

Aggregates the ROCs obtained from multiple sources into one global ROC.
Additionally, it's also possible to calculate the precision-recall curve.

## Usage

Install the package using one of the following options:

- pip3: `pip3 install roc-aggregator`
- this repository `pip3 install .`

Example:

- Obtain the global ROC curve from different sources by providing the false positive rate (fpr), true positive rate (tpr), thresholds (thresh), the total number of negative samples, and the total number of samples from each source:

```python
fpr, tpr, thresh_stack = roc_curve(
    [fpr_1, fpr_2],
    [tpr_1, tpr_2],
    [thresh_1, thresh_2],
    [negative_count_1, negative_count_2],
    [total_count_1, total_count_2]
)
```

- Calculate the AUC using numpy:
  
```python
np.trapz(tpr, fpr)
```

A complete example of the usage of the roc-aggregator can be found at `/roc-aggregator/examples/example.py`.

### Visualization

```python
plt.style.use('seaborn')
plt.plot(fpr, tpr, color=color, label=label, linestyle=linestyle)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.savefig('ROC',dpi=300)
plt.show()
```

## Testing

Unit tests are available at `/roc-aggregator/tests`.
Install the dependencies required and run the tests using `pytest` or `python3 setup.py test`.
