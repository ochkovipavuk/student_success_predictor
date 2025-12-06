import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture


def save_labels_and_model(labels: np.ndarray, model, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"cluster": labels}).to_csv(
        os.path.join(out_dir, "labels.csv"), index=False
    )
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)


def run_all_clustering(X_scaled, out_dir="clusters", n_clusters=3):
    """Run KMeans, GMM, Agglomerative, DBSCAN and save results under out_dir//"""
    # Ensure out dirs
    methods = {}

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    klabels = kmeans.fit_predict(X_scaled)
    save_labels_and_model(klabels, kmeans, os.path.join(out_dir, "kmeans"))
    methods["kmeans"] = klabels

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    glabels = gmm.fit_predict(X_scaled)
    save_labels_and_model(glabels, gmm, os.path.join(out_dir, "gmm"))
    methods["gmm"] = glabels

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    alabels = agg.fit_predict(X_scaled)
    save_labels_and_model(alabels, agg, os.path.join(out_dir, "agglomerative"))
    methods["agglomerative"] = alabels

    # DBSCAN (params can be tuned)
    dbscan = DBSCAN(eps=1.5, min_samples=10)
    dlabels = dbscan.fit_predict(X_scaled)
    save_labels_and_model(dlabels, dbscan, os.path.join(out_dir, "dbscan"))
    methods["dbscan"] = dlabels

    return methods
