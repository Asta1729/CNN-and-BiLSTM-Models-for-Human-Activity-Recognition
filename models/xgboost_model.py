from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pandas as pd
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.evaluate import evaluate_model

def load_data_xgb():
    base_path = os.path.join(ROOT_DIR, "data", "UCI HAR Dataset")
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    features = pd.read_csv(os.path.join(base_path, "features.txt"), sep="\s+", header=None, names=["index", "feature"])["feature"].tolist()
    
    # Load training data
    X_train = pd.read_csv(os.path.join(train_path, "X_train.txt"), sep="\s+", header=None)
    X_train.columns = features
    y_train = pd.read_csv(os.path.join(train_path, "y_train.txt"), sep="\s+", header=None, names=["Activity"])
    
    # Load test data
    X_test = pd.read_csv(os.path.join(test_path, "X_test.txt"), sep="\s+", header=None)
    X_test.columns = features
    y_test = pd.read_csv(os.path.join(test_path, "y_test.txt"), sep="\s+", header=None, names=["Activity"])

    
    y_train["Activity"] = y_train["Activity"] - 1
    y_test["Activity"]  = y_test["Activity"] - 1

    return X_train, X_test, y_train["Activity"], y_test["Activity"]


def train_xgboost():
    X_train, X_test, y_train, y_test = load_data_xgb()

    
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    X_train = X_train.values
    X_test  = X_test.values

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)   

def evaluate_xgboost():
    X_train, X_test, y_train, y_test = load_data_xgb()

    # convert to numpy arrays
    X_train = X_train.values
    X_test = X_test.values

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, "XGBoost")

if __name__ == "__main__":
    train_xgboost()
