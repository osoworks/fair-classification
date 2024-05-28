from __future__ import division
import numpy as np
import os, sys, traceback
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle


def get_classifier(clf_name):
    if clf_name == "logreg":
        return LogisticRegression()
    elif clf_name == "svm_linear":
        return SVC(kernel='linear', probability=True)
    else:
        raise ValueError("Unknown classifier name")


def get_fair_metrics(y_true, y_pred, sensitive_features):
    acc = accuracy_score(y_true, y_pred)

    unique_sensitive_values = np.unique(sensitive_features)
    metrics = {}

    for value in unique_sensitive_values:
        idx = sensitive_features == value
        tp = np.sum((y_true[idx] == 1) & (y_pred[idx] == 1))
        fp = np.sum((y_true[idx] == 0) & (y_pred[idx] == 1))
        tn = np.sum((y_true[idx] == 0) & (y_pred[idx] == 0))
        fn = np.sum((y_true[idx] == 1) & (y_pred[idx] == 0))

        metrics[value] = {
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "TNR": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0
        }

    return acc, metrics


if __name__ == "__main__":
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], size=100)
    x_sensitive = np.random.choice([0, 1], size=100)

    clf_name = "logreg"  # Change to "svm_linear" if you want to test SVM
    clf = get_classifier(clf_name)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    acc, metrics = get_fair_metrics(y, y_pred, x_sensitive)

    print("\n\n\nAccuracy: %0.3f\n" % acc)
    for key in metrics:
        print("Sensitive attribute value: %s" % key)
        print("TPR: %0.3f" % metrics[key]["TPR"])
        print("FPR: %0.3f" % metrics[key]["FPR"])
        print("TNR: %0.3f" % metrics[key]["TNR"])
        print("FNR: %0.3f\n" % metrics[key]["FNR"])