"""Segmentation (k-means++), PCA et détection d'anomalies (Isolation Forest)."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def standardize(df):
    return StandardScaler().fit_transform(df.values)


def pca_reduce(X, var_target=0.90):
    full = PCA().fit(X)
    n = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), var_target) + 1)
    p = PCA(n_components=n).fit(X)
    return p.transform(X), p


def select_k(X, k_min=2, k_max=10, seed=42):
    """Coude (inertie), silhouette, Calinski-Harabasz, Davies-Bouldin pour chaque k."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=seed).fit(X)
        rows.append({"k": k, "inertie": km.inertia_, "silhouette": silhouette_score(X, km.labels_),
                     "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_)})
    return pd.DataFrame(rows).set_index("k")


def run_kmeans(X, k, seed=42):
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=seed).fit(X)
    return km.labels_, km


def kmeans_outliers(X, km, quantile=0.95):
    """Titres les plus éloignés de leur centre : points 'isolés' par k-means (distance > quantile)."""
    d = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    return d > np.quantile(d, quantile), d


def isolation_forest(X, contamination=0.05, seed=42):
    """Retourne (drapeau anomalie, score) ; score plus élevé = plus atypique."""
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    return iso.predict(X) == -1, -iso.score_samples(X)
