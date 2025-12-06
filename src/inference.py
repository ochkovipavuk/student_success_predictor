import os
import pickle
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

from src.evaluate import evaluate_metrics


def _build_states_map(model) -> Dict[str, List[str]]:
    """Build a mapping var -> allowed state names from model CPDs."""
    states_map = {}
    for var in model.nodes():
        try:
            cpd = model.get_cpds(var)
            if cpd is None:
                continue
            states_map[var] = cpd.state_names[var]
        except Exception:
            # if CPD missing or not accessible, skip
            continue
    return states_map


def fix_evidence_types(evidence: dict, states_map: Dict[str, List[str]]) -> dict:
    """
    Try to coerce evidence values to strings that exist in states_map[var].
    This mirrors logic used in the notebook: try "float -> str", then int, else fallback.
    """
    fixed = {}
    for var, val in evidence.items():
        if var not in states_map:
            # unknown variable in states map — skip or store raw
            fixed[var] = str(val)
            continue

        allowed = states_map[var]
        # try direct match
        sval = str(val)
        if sval in allowed:
            fixed[var] = sval
            continue

        # try float representation (pgmpy sometimes stores floats as '0.0')
        try:
            svalf = str(float(val))
            if svalf in allowed:
                fixed[var] = svalf
                continue
        except Exception:
            pass

        # try int representation
        try:
            svali = str(int(float(val)))
            if svali in allowed:
                fixed[var] = svali
                continue
        except Exception:
            pass

        # fallback: pick nearest allowed (best-effort) by index if allowed looks numeric
        # else just use first allowed state
        chosen = allowed[0]
        fixed[var] = chosen
    return fixed


def run_inference(
    bayes_dir: str,
    clusters_dir: str,
    method: str,
    X_bayes: pd.DataFrame,
    out_dir: str = "inference_outputs",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    """
    Run inference using saved Bayesian network model (.bif) located at bayes_dir//model.bif.
    - Loads cluster labels from clusters_dir//labels.csv and attaches as 'target'
    - Splits data with sklearn.train_test_split (same seed as training)
    - Runs VariableElimination for each test row, computes expected numeric prediction
    (weighted average over states), and collects true code (index of state's label)
    - Saves predictions to CSV and returns (true_values, predictions, metrics)
    """
    # paths
    bif_path = os.path.join(bayes_dir, method, "model.bif")
    struct_path = os.path.join(bayes_dir, method, "structure.pkl")
    labels_path = os.path.join(clusters_dir, method, "labels.csv")

    if not os.path.exists(bif_path):
        raise FileNotFoundError(f"Bayes model .bif not found: {bif_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Cluster labels not found: {labels_path}")

    # load labels and attach target
    clusters = pd.read_csv(labels_path)["cluster"]
    df_bayes = X_bayes.copy()
    df_bayes["target"] = clusters.values

    # train/test split (must match training split to allow direct comparison)
    train_data, test_data = train_test_split(
        df_bayes, test_size=test_size, random_state=random_state
    )

    # load model
    model = DiscreteBayesianNetwork()
    # pgmpy provides model.load that returns the model
    model = model.load(bif_path)

    infer = VariableElimination(model)

    # build states_map
    states_map = _build_states_map(model)

    predictions = []
    true_values = []
    skipped = 0
    rows_info = []

    # Determine variables used for evidence (all nodes except target)
    relevant_vars = [v for v in model.nodes() if v != "target"]

    for idx, row in test_data.iterrows():
        # prepare evidence from row (only relevant vars)
        evidence_raw = row[relevant_vars].to_dict()
        evidence = fix_evidence_types(evidence_raw, states_map)

        try:
            q = infer.query(variables=["target"], evidence=evidence)
        except Exception as e:
            # inference failed for this row; skip
            skipped += 1
            rows_info.append({"idx": int(idx), "error": str(e)})
            continue

        # get names and probs
        try:
            states = q.state_names["target"]
            probs = q.values.flatten()
        except Exception as e:
            skipped += 1
            rows_info.append({"idx": int(idx), "error": f"bad query result: {e}"})
            continue

        if np.isnan(probs).any():
            skipped += 1
            rows_info.append({"idx": int(idx), "error": "NaN in probs"})
            continue

        # numeric expected value: assign codes [0..n_states-1]
        state_codes = np.arange(len(states))
        pred = float(np.sum(state_codes * probs))

        # compute true_code: try to find index of row['target'] in states
        true_raw = row["target"]
        found = None
        for try_val in (
            str(int(true_raw)) if not pd.isna(true_raw) else None,
            str(float(true_raw)) if not pd.isna(true_raw) else None,
            str(true_raw),
        ):
            if try_val is None:
                continue
            try:
                found = list(states).index(try_val)
                break
            except Exception:
                continue
        if found is None:
            # try any cast
            try:
                found = int(true_raw)
                if found not in list(range(len(states))):
                    # clamp
                    found = max(0, min(found, len(states) - 1))
            except Exception:
                # skip if cannot map
                skipped += 1
                rows_info.append(
                    {
                        "idx": int(idx),
                        "error": f"cannot map true target {true_raw} to states {states}",
                    }
                )
                continue

        predictions.append(pred)
        true_values.append(int(found))

    # Evaluate
    metrics = evaluate_metrics(true_values, predictions)

    # ensure output dir
    os.makedirs(out_dir, exist_ok=True)

    # Save predictions dataframe
    pred_df = pd.DataFrame(
        {
            "true": true_values,
            "pred_expected": predictions,
            "pred_round": np.round(predictions).astype(int),
        }
    )
    pred_df.to_csv(os.path.join(out_dir, f"predictions_{method}.csv"), index=False)

    # Save metadata about skipped rows
    if rows_info:
        pd.DataFrame(rows_info).to_csv(
            os.path.join(out_dir, f"skipped_rows_{method}.csv"), index=False
        )

    print(f"Inference finished. Rows processed: {len(predictions)}, skipped: {skipped}")
    # print("Metrics:", metrics)

    return true_values, predictions, metrics
