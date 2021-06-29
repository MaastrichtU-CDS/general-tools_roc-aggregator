import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn import metrics

from roc_aggregator import roc_curve, precision_recall_curve

def individual_roc(X, Y, model):
    pred_prob = model.predict_proba(X)

    fpr, tpr, thresh = metrics.roc_curve(Y, pred_prob[:,1], pos_label=1, drop_intermediate=False)
    auc_score = metrics.roc_auc_score(Y, pred_prob[:,1])
    prec, recall, thresh_prec = metrics.precision_recall_curve(Y, pred_prob[:,1], pos_label=1)
    prec_score = average_precision_score(Y, pred_prob[:,1])

    cm_by_threshold = []
    for th in thresh_prec:
        cm_by_threshold.append(confusion_matrix(Y, pred_prob[:,1] >= th, labels=[0,1]).ravel())

    return fpr, tpr, thresh, auc_score, prec, recall, thresh_prec, cm_by_threshold

def plot_roc(fpr, tpr, label, color, linestyle):
    plt.style.use('seaborn')
    plt.plot(fpr, tpr, color=color, label=label, linestyle=linestyle)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    #plt.savefig('ROC',dpi=300)

# 3 nodes
X1, y1 = make_classification(n_samples=500, n_classes=2, n_features=15, random_state=27)
X2, y2 = make_classification(n_samples=890, n_classes=2, n_features=15, random_state=22)
X3, y3 = make_classification(n_samples=321, n_classes=2, n_features=15, random_state=29)

# Complete dataset
X = np.concatenate((X1, X2, X3))
Y = np.concatenate((y1, y2, y3))

# Create global model
X_train, X_test, y_train, y_test = train_test_split(
    np.concatenate((X1, X2)),
    np.concatenate((y1, y2)),
    test_size=0.3,
    random_state=4
)

model = LogisticRegression()
model.fit(X_train, y_train)

# ROC curve with the full dataset
fpr_c, tpr_c, thresh_c, auc_score_c, prec_c, recall_c, thresh_pre_c, cm_c = individual_roc(X, Y, model)

# Calculate the ROC for each node
fpr_1, tpr_1, thresh_1, auc_score_1, _, _, thresh_pre_1, cm1 = individual_roc(X1, y1, model)
fpr_2, tpr_2, thresh_2, auc_score_2, _, _, thresh_pre_2, cm2 = individual_roc(X2, y2, model)
fpr_3, tpr_3, thresh_3, auc_score_3, _, _, thresh_pre_3, cm3 = individual_roc(X3, y3, model)

# Compute the global ROC
fpr, tpr, thresh_stack = roc_curve(
    [fpr_1, fpr_2, fpr_3],
    [tpr_1, tpr_2, tpr_3],
    [thresh_1, thresh_2, thresh_3],
    [np.count_nonzero(pred == 0) for pred in [y1, y2, y3]],
    [len(dataset) for dataset in [X1, X2, X3]]
)

plot_roc(fpr, tpr, 'roc-aggregator', 'orange', '--')
plot_roc(fpr_c, tpr_c, 'central case with sklearn', 'blue', 'dotted')
plt.show()

# Validate the results
assert np.array_equal(tpr[np.argmax(thresh_stack < 1) - 1:], tpr_c)
assert np.array_equal(fpr[np.argmax(thresh_stack < 1) - 1:], fpr_c)
print("Compare ROC AUC")
print(f'Central case with sklearn: {auc_score_c}')
print(f'roc-aggregator: {np.trapz(tpr, fpr)}')

# Compute the global precision-recall curve
pre, recall, thresh_stack = precision_recall_curve(
    [fpr_1, fpr_2, fpr_3],
    [tpr_1, tpr_2, tpr_3],
    [thresh_1, thresh_2, thresh_3],
    [np.count_nonzero(pred == 0) for pred in [y1, y2, y3]],
    [len(dataset) for dataset in [X1, X2, X3]]
)

plot_roc(pre, recall, 'roc-aggregator', 'orange', '--')
plot_roc(prec_c, recall_c, 'central case with sklearn', 'blue', 'dotted')
plt.show()

# Validate the results
print("Compare precision-recall curve")
print(f'Central case with sklearn: {np.trapz(recall_c, prec_c)}')
print(f'roc-aggregator: {np.trapz(recall, pre)}')
