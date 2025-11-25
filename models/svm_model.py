import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.evaluate import evaluate_model

def load_data_svm():
    base_path = os.path.join(ROOT_DIR, "data", "UCI HAR Dataset")

    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    features = pd.read_csv(os.path.join(base_path, "features.txt"), sep="\s+", header=None, names=["index", "feature"])["feature"].tolist()

    # Load training data
    X_train = pd.read_csv(os.path.join(train_path, "X_train.txt"), sep="\s+", header=None)
    X_train.columns = features
    y_train = pd.read_csv(os.path.join(train_path, "y_train.txt"), sep="\s+", header=None, names=["Activity"])["Activity"]
    
    # Load test data
    X_test = pd.read_csv(os.path.join(test_path, "X_test.txt"), sep="\s+", header=None)
    X_test.columns = features
    y_test = pd.read_csv(os.path.join(test_path, "y_test.txt"), sep="\s+", header=None, names=["Activity"])["Activity"]
    
    return X_train, X_test, y_train, y_test

def train_svm():
    X_train, X_test, y_train, y_test = load_data_svm()

    svm = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        decision_function_shape="ovr"
    )

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    return accuracy_score(y_test, y_pred)

def evaluate_svm():
    X_train, X_test, y_train, y_test = load_data_svm()
    svm = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        decision_function_shape="ovr"
    )
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    evaluate_model(y_test, y_pred, "Support Vector Machine (SVM)")

if __name__ == "__main__":
    train_svm()
