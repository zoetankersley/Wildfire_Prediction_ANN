# Baseline models
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def dummy_model(seed=42):
    """Two floor baselines: always guess the most common class, and guess each class with equal chance."""
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "dummy_uniform": DummyClassifier(strategy="uniform", random_state=seed),
    }


def linear_model(seed=42):
    """Logistic regression with L2 regularization. class_weight='balanced' makes rare classes count more."""
    return {
        "logistic": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
        ),
    }


def tree_model(seed=42):
    """Random forest: captures nonlinear patterns; the main non-neural-network comparison."""
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        ),
    }
