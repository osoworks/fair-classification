import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

def get_classifier(clf_name):
    if clf_name == "logreg":
        return LogisticRegression()
    elif clf_name == "svm_linear":
        return SVC(kernel='linear', probability=True)
    else:
        raise ValueError("Unknown classifier name")

def measure_final_score(X, clf, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    print(f"Accuracy: {accuracy}")
    print(f"AUC-ROC: {auc_roc}")
    print(f"Sensitivity (Recall): {sensitivity}")
    print(f"Specificity: {specificity}")
    
    return accuracy, auc_roc, sensitivity, specificity

def print_results(acc, auc, sens, spec):
    print("\nResults:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Sensitivity (Recall): {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print("\n")

if __name__ == "__main__":
    # 예제 데이터 생성
    np.random.seed(1234)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], size=100)
    x_sensitive = np.random.choice([0, 1], size=100)

    clf_name = "logreg"  # SVM을 테스트하려면 "svm_linear"로 변경
    clf = get_classifier(clf_name)

    clf.fit(X, y)
    y_pred = clf.predict(X)

    accuracy, auc, sensitivity, specificity = measure_final_score(X, clf, y, y_pred)

    print_results(accuracy, auc, sensitivity, specificity)