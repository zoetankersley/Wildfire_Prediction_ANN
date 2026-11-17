# Baseline Model
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
...
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
def dummy_model(seed=42):
    """Two floor baseline- the most common class and a guess based on the class proportions"""
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "dummy_uniform": DummyClassifier(strategy="uniform", random_state=seed)
    }
    
def linear_model(seed=42):
    """Logistic regression with L2 regularization"""
    return {
        "logistic": LogisticRegression(penalty="l2", random_state=seed, max_iter=1000)
    }