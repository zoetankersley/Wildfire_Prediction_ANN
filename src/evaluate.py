"""Wildfire size regression metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import to_acres


def regression_metrics(actual_log, predicted_log):
    """Report errors in log space and acres, clipping negative predicted acres."""
    actual_acres = to_acres(actual_log)
    predicted_acres = np.maximum(to_acres(predicted_log.astype(float)), 0)
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(actual_log, predicted_log))),
        "mae_log": float(mean_absolute_error(actual_log, predicted_log)),
        "r2_log": float(r2_score(actual_log, predicted_log)),
        "rmse_acres": float(np.sqrt(mean_squared_error(actual_acres, predicted_acres))),
        "mae_acres": float(mean_absolute_error(actual_acres, predicted_acres)),
    }
