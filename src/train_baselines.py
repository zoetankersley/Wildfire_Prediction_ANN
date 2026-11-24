"""Fit wildfire baselines on the same processed data and splits as the MLP."""

import argparse
import json
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)

from .data import (
    DATA_PATH, MODEL_NAMES, SPLIT_SEED, TARGET, experiment_features, load_experiment, log_target,
)
from .evaluate import regression_metrics
from .models import dummy_model, linear_model, tree_model


def classification_metrics(actual, predicted):
    """Overall scores, plus precision/recall/F1 for each class and the confusion matrix."""
    labels = sorted(set(actual) | set(predicted))
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "per_class": {
            label: {
                "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        # Rows are the true class, columns the predicted class, both in `labels` order
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        },
    }


def score_model(model, inputs, targets, metrics):
    x_train, x_val, x_test = inputs
    y_train, y_val, y_test = targets
    model.fit(x_train, y_train)
    return {
        "validation": metrics(y_val, model.predict(x_val)),
        "test": metrics(y_test, model.predict(x_test)),
    }


def run_baselines(data_path, classification_target=TARGET):
    df, parts, metadata = load_experiment(data_path)
    if classification_target not in df:
        raise ValueError(f"Missing classification column: {classification_target}")
    if any(part[classification_target].isna().any() for part in parts):
        raise ValueError(f"Missing {classification_target} labels in the shared splits")

    groups, inputs = experiment_features(df, parts, MODEL_NAMES)
    regression_targets = tuple(log_target(part).to_numpy() for part in parts)
    class_targets = tuple(part[classification_target].to_numpy() for part in parts)

    regression = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "dummy_median": DummyRegressor(strategy="median"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SPLIT_SEED,
        ),
    }
    classification = {
        "dummy_majority": dummy_model(SPLIT_SEED)["dummy_majority"],
        **linear_model(SPLIT_SEED),
        **tree_model(SPLIT_SEED),
    }
    return {
        **metadata,
        "regression_target": "log1p(FIRE_SIZE)",
        "classification_target": classification_target,
        "features": groups,
        "regression": {
            group: {
                name: score_model(model, x, regression_targets, regression_metrics)
                for name, model in regression.items()
            }
            for group, x in inputs.items()
        },
        "classification": {
            group: {
                name: score_model(model, x, class_targets, classification_metrics)
                for name, model in classification.items()
            }
            for group, x in inputs.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--classification-target", choices=(TARGET, "FIRE_SIZE_CLASS"), default=TARGET)
    parser.add_argument("--output", type=Path, help="Save JSON results instead of printing them")
    args = parser.parse_args()

    results = run_baselines(args.data, args.classification_target)
    report = json.dumps(results, indent=2, allow_nan=False)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
