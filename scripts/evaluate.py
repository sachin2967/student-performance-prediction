"""Compare pass/fail classifiers on held-out folds of UCI student-por.csv."""

import argparse
import json
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
COURSE_FILE = "student-por.csv"
SEED = 42
FOLDS = 5
LABELS = (0, 1)
CLASS_NAMES = ("fail", "pass")


def make_forest():
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()),
                make_column_selector(dtype_include="number"),
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                ),
                make_column_selector(dtype_include=["object", "string", "category"]),
            ),
        ]
    )
    return make_pipeline(
        preprocessing, RandomForestClassifier(n_estimators=200, random_state=SEED)
    )


def metrics(truth, predictions):
    precision, recall, _, support = precision_recall_fscore_support(
        truth, predictions, labels=LABELS, zero_division=0
    )
    return {
        "accuracy": accuracy_score(truth, predictions),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=LABELS
        ).tolist(),
        "classes": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def evaluate_threshold(data, pass_mark):
    truth = (data["G3"] >= pass_mark).astype(int)
    features = data.drop(columns="G3")
    configurations = {
        "majority_class": (DummyClassifier(strategy="most_frequent"), features),
        "random_forest": (make_forest(), features),
        "random_forest_without_g1_g2": (
            make_forest(),
            features.drop(columns=["G1", "G2"]),
        ),
    }
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(features, truth))
    results = {}
    for name, (model, inputs) in configurations.items():
        predictions = np.empty(len(truth), dtype=int)
        fold_results = []
        for fold, (train, test) in enumerate(splits, start=1):
            model.fit(inputs.iloc[train], truth.iloc[train])
            predictions[test] = model.predict(inputs.iloc[test])
            fold_results.append(
                {"fold": fold, **metrics(truth.iloc[test], predictions[test])}
            )
        results[name] = {
            "out_of_fold": metrics(truth, predictions),
            "folds": fold_results,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("eval/cross_validation.json")
    )
    args = parser.parse_args()

    with urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    with ZipFile(BytesIO(payload)) as archive:
        course = archive.read(COURSE_FILE)
    data = pd.read_csv(BytesIO(course), sep=";")
    report = {
        "command": ["python", *sys.argv],
        "dataset": {
            "url": DATA_URL,
            "file": COURSE_FILE,
            "sha256": sha256(course).hexdigest(),
            "rows": len(data),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "split": {
            "method": "StratifiedKFold",
            "folds": FOLDS,
            "shuffle": True,
            "seed": SEED,
        },
        "confusion_matrix_labels": CLASS_NAMES,
        "zero_division": 0,
        "results_by_pass_mark": {
            mark: evaluate_threshold(data, mark) for mark in (8, 10)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
