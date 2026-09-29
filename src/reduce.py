"""skalierung, gower-distanz und dimensionsreduktion mit pca, mds und lle."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, LocallyLinearEmbedding, trustworthiness
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale(X: pd.DataFrame, method: str) -> np.ndarray:
    """standard = z-werte, sonst minmax auf 0 bis 1."""
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    return scaler.fit_transform(X)


def gower_distance(X: pd.DataFrame) -> np.ndarray:
    """gower-distanz, bei numerisch kodierten merkmalen = mittlere auf die spannweite normierte absolute differenz."""
    values = X.to_numpy(dtype=float)
    ranges = values.max(axis=0) - values.min(axis=0)
    ranges[ranges == 0] = 1.0
    return squareform(pdist(values / ranges, metric="cityblock")) / values.shape[1]


def fit_pca(Z: np.ndarray, columns: list[str], threshold: float, random_state: int) -> dict:
    """volle pca, die anzahl komponenten ergibt sich aus der schwelle für die kumulierte varianz."""
    pca = PCA(random_state=random_state).fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_comp = int(np.searchsorted(cum, threshold) + 1)
    loadings = pd.DataFrame(pca.components_.T, index=columns,
                            columns=[f"PC{i + 1}" for i in range(len(evr))])
    return {
        "model": pca,
        "scores": pca.transform(Z),
        "evr": evr,
        "cum": cum,
        "n_components": n_comp,
        "eigenvalues": pca.explained_variance_,
        "loadings": loadings,
    }


def kruskal_stress(D: np.ndarray, Y: np.ndarray) -> float:
    """stress-1 nach kruskal (1964), normiert über die distanzen in der einbettung."""
    d = squareform(pdist(Y))
    iu = np.triu_indices_from(D, k=1)
    return float(np.sqrt(((D[iu] - d[iu]) ** 2).sum() / (d[iu] ** 2).sum()))


def fit_mds(D: np.ndarray, n_components: int, cfg_red: dict, random_state: int) -> tuple[np.ndarray, float]:
    """metrische mds (smacof) auf einer fertigen distanzmatrix."""
    mds = MDS(n_components=n_components, metric_mds=True, metric="precomputed",
              n_init=cfg_red["mds_n_init"], max_iter=cfg_red["mds_max_iter"],
              init="random", random_state=random_state)
    Y = mds.fit_transform(D)
    return Y, kruskal_stress(D, Y)


def fit_lle(Z: np.ndarray, n_neighbors: int, reg: float, random_state: int) -> tuple[np.ndarray, float]:
    """lle in 2d, gibt einbettung und rekonstruktionsfehler zurück."""
    lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2, reg=reg,
                                 eigen_solver="dense", random_state=random_state)
    Y = lle.fit_transform(Z)
    return Y, float(lle.reconstruction_error_)


def run_reduction(X_sel: pd.DataFrame, Z: np.ndarray, cfg: dict) -> dict:
    """pca, mds und lle rechnen und die 2d-einbettungen per trustworthiness vergleichen."""
    red, rs = cfg["reduction"], cfg["random_state"]
    res: dict = {}

    t0 = time.perf_counter()
    res["pca"] = fit_pca(Z, list(X_sel.columns), red["pca_variance_threshold"], rs)
    res["runtime"] = {"PCA": time.perf_counter() - t0}

    D = gower_distance(X_sel)
    res["gower"] = D
    res["mds_stress"] = {}
    for dim in red["mds_dims"]:
        t0 = time.perf_counter()
        Y, stress = fit_mds(D, dim, red, rs)
        res["mds_stress"][dim] = stress
        if dim == 2:
            res["mds_2d"] = Y
            res["runtime"]["MDS"] = time.perf_counter() - t0

    res["lle_error"], res["lle_embeddings"], lle_time, lle_tw = {}, {}, {}, {}
    for k in red["lle_neighbors"]:
        t0 = time.perf_counter()
        Y, err = fit_lle(Z, k, red["lle_reg"], rs)
        lle_time[k] = time.perf_counter() - t0
        res["lle_error"][k] = err
        res["lle_embeddings"][k] = Y
        lle_tw[k] = trustworthiness(Z, Y, n_neighbors=red["trust_select_k"])
    # für den vergleich die lle-variante mit der besten trustworthiness nehmen
    best_k = max(lle_tw, key=lle_tw.get)
    res["lle_best_neighbors"] = best_k
    res["runtime"]["LLE"] = lle_time[best_k]

    emb = {"PCA": res["pca"]["scores"][:, :2], "MDS": res["mds_2d"],
           "LLE": res["lle_embeddings"][best_k]}
    res["embeddings_2d"] = emb
    tw = {}
    for name, Y in emb.items():
        for k in red["trustworthiness_k"]:
            tw[f"{name}_k{k}"] = trustworthiness(Z, Y, n_neighbors=k)
    for k in red["trustworthiness_k"]:
        tw[f"MDS_gower_k{k}"] = trustworthiness(D, res["mds_2d"], n_neighbors=k, metric="precomputed")
    for k_lle, v in lle_tw.items():
        tw[f"LLE_nn{k_lle}_k{red['trust_select_k']}"] = v
    res["trustworthiness"] = tw
    return res
