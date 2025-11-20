import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.evaluate import evaluate_model

def load_data_rf():
    base_path = os.path.join(ROOT_DIR, "data", "UCI HAR Dataset")
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    # Load feature names
    features = pd.read_csv(os.path.join(base_path, "features.txt"), sep="\s+", header=None, names=["index", "feature"])["feature"].tolist()

    # Load train data
    X_train = pd.read_csv(os.path.join(train_path, "X_train.txt"), sep="\s+", header=None)
    X_train.columns = features
    y_train = pd.read_csv(os.path.join(train_path, "y_train.txt"), sep="\s+", header=None, names=["Activity"])

    # Load test data
    X_test = pd.read_csv(os.path.join(test_path, "X_test.txt"), sep="\s+", header=None)
    X_test.columns = features
    y_test = pd.read_csv(os.path.join(test_path, "y_test.txt"), sep="\s+", header=None, names=["Activity"])

    return X_train, X_test, y_train, y_test


def train_random_forest():
    X_train, X_test, y_train, y_test = load_data_rf()

    rf = RandomForestClassifier(
    n_estimators=351,        
    max_depth=30,
    max_features= 'log2',     
    min_samples_split=8,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1)

    rf.fit(X_train, y_train.values.ravel())
    y_pred = rf.predict(X_test)
    evaluate_model(y_test, y_pred, "Random Forest")

if __name__ == "__main__":
    train_random_forest()
