"""clustering mit k-means, gmm und ward, dazu gütemaße und bootstrap-stabilität."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture


def fit_model(method: str, X: np.ndarray, k: int, cfg: dict, cov: str = "full"):
    """passt ein clustering-modell an und gibt labels und modell zurück."""
    rs, c = cfg["random_state"], cfg["clustering"]
    if method == "kmeans":
        m = KMeans(n_clusters=k, init="k-means++", n_init=c["kmeans_n_init"], random_state=rs).fit(X)
        return m.labels_, m
    if method == "gmm":
        m = GaussianMixture(n_components=k, covariance_type=cov, n_init=c["gmm_n_init"], random_state=rs).fit(X)
        return m.predict(X), m
    if method == "ward":
        m = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        return m.labels_, m
    raise ValueError(method)


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """silhouette, davies-bouldin, calinski-harabasz und anteil des kleinsten clusters."""
    shares = np.bincount(labels) / len(labels)
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "min_share": float(shares.min()),
    }


def scan_k(X: np.ndarray, cfg: dict, methods: list[str], X_ref: np.ndarray | None = None) -> pd.DataFrame:
    """gütemaße für alle verfahren und k aus k_range, dazu wss bei k-means und bic bei gmm.

    mit X_ref wird die silhouette zusätzlich in diesem gemeinsamen raum berechnet, damit
    lösungen aus verschiedenen merkmalsräumen vergleichbar sind.
    """
    k_lo, k_hi = cfg["clustering"]["k_range"]
    rows = []
    for k in range(k_lo, k_hi + 1):
        for method in methods:
            if method == "gmm":
                for cov in cfg["clustering"]["gmm_covariance_types"]:
                    labels, m = fit_model("gmm", X, k, cfg, cov)
                    # gmm kann alles einem cluster zuordnen, dann gibt es keine silhouette
                    if len(np.unique(labels)) < 2:
                        continue
                    row = {"verfahren": f"gmm_{cov}", "k": k, "bic": m.bic(X), **internal_metrics(X, labels)}
                    if X_ref is not None:
                        row["silhouette_ref"] = silhouette_score(X_ref, labels)
                    rows.append(row)
            else:
                labels, m = fit_model(method, X, k, cfg)
                row = {"verfahren": method, "k": k, **internal_metrics(X, labels)}
                if method == "kmeans":
                    row["wss"] = m.inertia_
                if X_ref is not None:
                    row["silhouette_ref"] = silhouette_score(X_ref, labels)
                rows.append(row)
    return pd.DataFrame(rows)


def gmm_bic_scan(X: np.ndarray, cfg: dict) -> pd.DataFrame:
    """bic der gmm je kovarianztyp über den erweiterten suchbereich gmm_bic_k_range."""
    k_lo, k_hi = cfg["clustering"]["gmm_bic_k_range"]
    rows = []
    for cov in cfg["clustering"]["gmm_covariance_types"]:
        for k in range(k_lo, k_hi + 1):
            _, m = fit_model("gmm", X, k, cfg, cov)
            rows.append({"kovarianz": cov, "k": k, "bic": m.bic(X)})
    return pd.DataFrame(rows)


def describe_bic(bic: pd.DataFrame) -> dict:
    """je kovarianztyp: k mit kleinstem bic, ob das minimum am rand liegt und wie unruhig der verlauf ist."""
    out = {}
    for cov, d in bic.groupby("kovarianz"):
        d = d.sort_values("k")
        b, ks = d.bic.to_numpy(), d.k.to_numpy()
        i = int(np.argmin(b))
        steps = np.diff(b)
        # schritte gegen den trend: anstiege vor dem minimum und abfälle danach (bei einem klaren minimum 0)
        against = int((steps[:i] > 0).sum() + (steps[i:] < 0).sum())
        out[cov] = {"bestes_k": int(ks[i]), "bic_min": float(b[i]), "minimum_am_rand": bool(i in (0, len(b) - 1)),
                    "schritte_gegen_trend": against}
    return out


def _assign_nearest_centroid(X: np.ndarray, Xb: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """jeden punkt dem nächsten zentroid der bootstrap-lösung zuordnen (ward hat kein predict)."""
    centroids = np.vstack([Xb[labels_b == j].mean(axis=0) for j in np.unique(labels_b)])
    return np.argmin(((X[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)


def bootstrap_stability(X: np.ndarray, method: str, k: int, cfg: dict, cov: str = "full") -> np.ndarray:
    """stabilität in anlehnung an hennig (2007): modell auf resample, alle punkte zuordnen, ari zur referenz."""
    ref, _ = fit_model(method, X, k, cfg, cov)
    rng = np.random.default_rng(cfg["random_state"])
    aris = []
    for _ in range(cfg["clustering"]["bootstrap_resamples"]):
        idx = rng.integers(0, len(X), len(X))
        Xb = X[idx]
        labels_b, m = fit_model(method, Xb, k, cfg, cov)
        if method in ("kmeans", "gmm"):
            pred = m.predict(X)
        else:
            pred = _assign_nearest_centroid(X, Xb, labels_b)
        aris.append(adjusted_rand_score(ref, pred))
    return np.array(aris)


def ari_matrix(label_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    """paarweiser ari zwischen mehreren clusterlösungen."""
    names = list(label_sets)
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            M.loc[a, b] = adjusted_rand_score(label_sets[a], label_sets[b])
    return M


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """cluster nach größe neu nummerieren (größtes = 0), damit die beschriftung stabil bleibt."""
    order = pd.Series(labels).value_counts().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[l] for l in labels])
