import os
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


def load_labels(method, clusters_dir="clusters"):
    path = os.path.join(clusters_dir, method, "labels.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)["cluster"].values


def compare_silhouette(
    X_scaled, X_cluster_df, methods=None, clusters_dir="clusters", show=False
):
    methods = methods or ["kmeans", "gmm", "agglomerative", "dbscan"]
    scores = {}
    for m in methods:
        labels = load_labels(m, clusters_dir)
        if labels is None:
            scores[m] = None
            continue
        if len(np.unique(labels)) < 2:
            scores[m] = None
            continue
        try:
            scores[m] = silhouette_score(X_scaled, labels)
        except Exception:
            scores[m] = None
    # print and plot simple bar
    print("Silhouette scores:")
    for k, v in scores.items():
        print(f"{k}: {v}")
        clean = {k: v for k, v in scores.items() if v is not None}
    if clean:
        plt.figure(figsize=(8, 4))
        plt.bar(list(clean.keys()), list(clean.values()))
        plt.title("Silhouette score comparison")
        plt.savefig(
            os.path.join(clusters_dir, "plot_silhouette.png"),
            dpi=300,
            bbox_inches="tight",
        )
        if show:
            plt.show()
        else:
            plt.close()
    return scores


def plot_embeddings(X_scaled, clusters_dir="clusters", show=False):
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print("Explained variance (PCA):", pca.explained_variance_ratio_)

    # try UMAP if installed
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
    except Exception:
        X_umap = None

    methods = ["kmeans", "gmm", "agglomerative", "dbscan"]
    for m in methods:
        labels = load_labels(m, clusters_dir)
        if labels is None:
            continue
        plt.figure(figsize=(6, 4))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=8, cmap="tab10")
        plt.title(f"PCA — {m}")
        plt.savefig(
            os.path.join(clusters_dir, m, "plot_pca.png"), dpi=300, bbox_inches="tight"
        )
        if show:
            plt.show()
        else:
            plt.close()
        if X_umap is not None:
            plt.figure(figsize=(6, 4))
            plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=8, cmap="tab10")
            plt.title(f"UMAP — {m}")
            plt.savefig(
                os.path.join(clusters_dir, m, "plot_umap.png"),
                dpi=300,
                bbox_inches="tight",
            )
            if show:
                plt.show()
            else:
                plt.close()


def cluster_analysis(df_common: pd.DataFrame, clusters_col="cluster"):
    """
    Compute basic cluster descriptive statistics and print summary.
    """
    common_df = df_common.copy()
    if clusters_col not in common_df.columns:
        raise ValueError(f"{clusters_col} not in dataframe")
    numeric_cols = common_df.select_dtypes(include=[np.number]).columns.tolist()
    if clusters_col in numeric_cols:
        numeric_cols.remove(clusters_col)

    print("Cluster sizes:")
    print(common_df[clusters_col].value_counts())

    # median representative per cluster for numeric cols
    reps = {}
    for c in sorted(common_df[clusters_col].unique()):
        sub = common_df[common_df[clusters_col] == c]
        rep = {col: sub[col].median() for col in numeric_cols}
        reps[c] = rep

    print("Numeric medians per cluster (sample):")
    for c, rep in reps.items():
        print(
            f"Cluster {c}: { {k:v for k,v in rep.items() if k in ['age','current_cgpa','attendance_pct'] and k in rep} }"
        )
    return reps


def plot_bayes_network(
    model_dir="bayes_nets", method="gmm", out_dir="bayes_nets", show=False
):
    """
    Visualizes the structure of the Bayesian network and saves the image.
    """

    with open(os.path.join(model_dir, method, "structure.pkl"), "rb") as f:
        model = pickle.load(f)

    save_path = os.path.join(out_dir, method, "graph_structure.png")

    # Создаём каталоги при необходимости
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Преобразуем модель PGMPY → обычный граф NetworkX
    G = nx.DiGraph()
    # G.add_nodes_from(model.nodes())
    G.add_edges_from(model)

    # Настраиваем расположение узлов
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.2, seed=42)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=1800,
        arrowsize=20,
        font_size=10,
    )

    """pos = nx.spring_layout(G, k=1.5, seed=42)

    # Узлы
    nx.draw_networkx_nodes(G, pos, node_color="#4C72B0", node_size=1400, alpha=0.9)

    # Рёбра
    nx.draw_networkx_edges(
        G, pos, edge_color="#222222", arrows=True, arrowsize=20, width=2
    )

    # Подписи узлов
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="white")"""

    plt.title("Bayesian Network Structure", fontsize=16)
    # plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Bayesian network graph saved to: {save_path}")


def data_correlation_matrix(
    clusters_dir: str,
    method: str,
    X_bayes: pd.DataFrame,
    out_dir: str = "clusters",
    show=False,
):
    labels_path = os.path.join(clusters_dir, method, "labels.csv")
    save_path = os.path.join(out_dir, method, "correlation_matrix.png")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Cluster labels not found: {labels_path}")

    # load labels and attach target
    clusters = pd.read_csv(labels_path)["cluster"]
    df_bayes = X_bayes.copy()
    df_bayes["target"] = clusters.values

    categorical_cols = df_bayes.select_dtypes(include=["object"]).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_bayes[col] = le.fit_transform(df_bayes[col].astype(str))
        label_encoders[col] = le

    correlation_matrix = df_bayes.corr()

    plt.figure(figsize=(16, 14))

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Correlation matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
