import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator
import logging

logging.getLogger("pgmpy").setLevel(logging.CRITICAL)


def train_bayes_network(
    X_bayes: pd.DataFrame, clusters_dir="clusters", method="gmm", out_dir="bayes_nets"
):
    """
    Train a discrete Bayesian network using cluster labels as target.
    Expects that clusters//labels.csv exists.
    """
    labels_path = os.path.join(clusters_dir, method, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    clusters = pd.read_csv(labels_path)["cluster"]
    df_bayes = X_bayes.copy()
    df_bayes["target"] = clusters

    # train/test split
    train_data, test_data = train_test_split(df_bayes, test_size=0.2, random_state=42)

    # structure learning
    hc = HillClimbSearch(train_data)
    best_model = hc.estimate(scoring_method=BIC(train_data))

    # initialize and fit
    model = DiscreteBayesianNetwork(best_model.edges())
    model.fit(train_data, estimator=MaximumLikelihoodEstimator)

    # save
    save_dir = os.path.join(out_dir, method)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "structure.pkl"), "wb") as f:
        pickle.dump(best_model.edges(), f)
    # save .bif
    model.save(os.path.join(save_dir, "model.bif"))

    return model, best_model.edges()
