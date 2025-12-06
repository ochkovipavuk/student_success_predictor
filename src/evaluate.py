from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    classification_report,
)


def evaluate_metrics(true: List[int], pred: List[float]) -> Dict[str, Any]:
    """
    Compute MAE, RMSE and Accuracy.
    Returns dictionary and prints human-readable summary.
    """
    if len(true) == 0:
        return {"error": "no data"}

    y_true = np.array(true)
    y_pred = np.array(pred)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # accuracy after rounding predictions to nearest int and clipping to valid range
    y_pred_round = np.rint(y_pred).astype(int)
    # optionally clip to valid class indices (0..max(true, predicted))
    max_class = max(int(y_true.max()), int(y_pred_round.max()))
    y_pred_round = np.clip(y_pred_round, 0, max_class)

    try:
        acc = float(accuracy_score(y_true, y_pred_round))
    except Exception:
        acc = None

    # simple classification report (only if small number unique)
    try:
        report = classification_report(
            y_true, y_pred_round, zero_division=0, output_dict=True
        )
    except Exception:
        report = None

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "accuracy": acc,
        "classification_report": report,
    }

    # print summary
    print("Evaluation summary:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    if acc is not None:
        print(f"  Accuracy: {acc:.4f}")
    """if report is not None:
        print("  --- classification report keys:", list(report.keys())[:5])"""

    return metrics
