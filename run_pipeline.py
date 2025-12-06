# run_pipeline.py

"""
Run the full pipeline (minimal).
Запускает: загрузку данных -> предобработка -> кластеризация -> визуализации -> байесова сеть.
"""
import os
from src.data_prep import load_data, clean_data, prepare_cluster_and_bayes
from src.clustering import run_all_clustering
from src.visualize import (
    compare_silhouette,
    plot_embeddings,
    cluster_analysis,
    plot_bayes_network,
    data_correlation_matrix,
)
from src.bayes_net import train_bayes_network
from src.inference import run_inference

import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "data/Students_Performance_data_set.xlsx"
CLUSTERS_DIR = "clusters"
BAYES_DIR = "bayes_nets"
INFERENCE_OUT = "inference_outputs"
METHOD_BAYES = "gmm"


def main():
    os.makedirs("clusters", exist_ok=True)
    os.makedirs("bayes_nets", exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Cleaning data...")
    df, num_cols, cat_cols = clean_data(df)

    print("Preparing cluster and bayes datasets...")
    X_cluster_df, X_scaled, encoder_scaler_dict, X_bayes = prepare_cluster_and_bayes(
        df, num_cols, cat_cols
    )

    print("Running clustering...")
    run_all_clustering(X_scaled, out_dir=CLUSTERS_DIR, n_clusters=3)

    print("Comparing silhouettes...")
    compare_silhouette(
        X_scaled,
        X_cluster_df,
        methods=["kmeans", "gmm", "agglomerative", "dbscan"],
        clusters_dir=CLUSTERS_DIR,
    )

    print("Plot embeddings (PCA + UMAP)...")
    plot_embeddings(X_scaled, clusters_dir=CLUSTERS_DIR)

    data_correlation_matrix(CLUSTERS_DIR, METHOD_BAYES, X_bayes, out_dir=CLUSTERS_DIR)

    print("Train Bayesian network...")
    # load labels for chosen method inside train_bayes_network
    train_bayes_network(
        X_bayes, clusters_dir=CLUSTERS_DIR, method=METHOD_BAYES, out_dir=BAYES_DIR
    )
    plot_bayes_network()

    print("Inference Bayesian network...")
    true_values, predictions, metrics = run_inference(
        bayes_dir=BAYES_DIR,
        clusters_dir=CLUSTERS_DIR,
        method=METHOD_BAYES,
        X_bayes=X_bayes,
        out_dir=INFERENCE_OUT,
    )

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
