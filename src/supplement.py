"""ergänzende analysen: kreuztabellen, sensitivität des endmodells, nicht-metrische mds und stichprobenübersicht."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.manifold import MDS, trustworthiness
from sklearn.metrics import adjusted_rand_score

from src import cluster, reduce


def crosstab_with_sums(a: np.ndarray, b: np.ndarray, name_a: str, name_b: str) -> pd.DataFrame:
    """kreuztabelle zweier lösungen (cluster ab 1) mit zeilen- und spaltensummen."""
    ct = pd.crosstab(pd.Series(a + 1, name=name_a), pd.Series(b + 1, name=name_b),
                     margins=True, margins_name="Summe")
    ct.index.name = f"{name_a} (Zeilen) × {name_b} (Spalten)"
    return ct


def refit_kmeans(X: pd.DataFrame, cfg: dict, k: int) -> tuple[np.ndarray, int]:
    """gleiche schritte wie im endmodell: skalieren, pca bis zur varianzschwelle, k-means."""
    Z = reduce.scale(X, cfg["scaling"]["method"])
    p = reduce.fit_pca(Z, list(X.columns), cfg["reduction"]["pca_variance_threshold"], cfg["random_state"])
    labels, _ = cluster.fit_model("kmeans", p["scores"][:, : p["n_components"]], k, cfg)
    return labels, p["n_components"]


def dont_know_matrix(df: pd.DataFrame, X_sel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """„weiß nicht“ als eigene indikatorvariable statt als 0,5 kodieren.

    der itemwert einer weiß-nicht-antwort wird durch das itemmittel der übrigen antworten ersetzt und trägt
    so keine information mehr, die information steckt nur im indikator. der unsicherheitsindex entfällt,
    weil er genau der mittelwert dieser indikatoren ist.
    """
    X = X_sel.drop(columns="uncertainty_index").copy()
    for col in cfg["uncertainty_items"]:
        dk = df[col].isin(cfg["uncertain_answers"]).to_numpy()
        # fehlende (nicht zutreffende) antworten bleiben wie im endmodell bei 0,5
        known = ~dk & df[col].notna().to_numpy()
        X.loc[dk, col] = X.loc[known, col].mean()
        X[f"dk_{col}"] = dk.astype(float)
    return X


def sensitivity(df: pd.DataFrame, X_sel: pd.DataFrame, labels: np.ndarray, cfg: dict) -> dict:
    """endmodell unter abgewandelten bedingungen neu rechnen und per ari mit der hauptlösung vergleichen."""
    k = cfg["clustering"]["final_k"]
    out = {}
    us = (df["country_work"] == "United States of America").to_numpy()
    for name, m in (("nur_usa", us), ("ohne_usa", ~us)):
        lab, n_hk = refit_kmeans(X_sel[m], cfg, k)
        out[name] = {"n": int(m.sum()), "pca_n_komponenten": n_hk,
                     "ari_zur_hauptloesung": adjusted_rand_score(labels[m], lab)}
    variants = {"ohne_mh_benefits": X_sel.drop(columns="mh_benefits"),
                "weiss_nicht_als_indikator": dont_know_matrix(df, X_sel, cfg)}
    for name, X in variants.items():
        lab, n_hk = refit_kmeans(X, cfg, k)
        lab = cluster.relabel_by_size(lab)
        ct = pd.crosstab(lab + 1, labels + 1)
        out[name] = {"n_merkmale": X.shape[1], "pca_n_komponenten": n_hk,
                     "ari_zur_hauptloesung": adjusted_rand_score(labels, lab),
                     "clustergroessen": np.bincount(lab).tolist(),
                     # zeilen = cluster der variante, spalten = cluster des endmodells
                     "kreuztabelle": {int(r): {int(c): int(v) for c, v in row.items()}
                                      for r, row in ct.to_dict(orient="index").items()},
                     "hoechste_varianz": X.var().sort_values(ascending=False).head(5).round(4).to_dict()}
    return out


def nonmetric_mds(D: np.ndarray, Z: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """nicht-metrische mds (smacof) in 2d auf der gower-distanz, stress-1 und trustworthiness."""
    red = cfg["reduction"]
    mds = MDS(n_components=2, metric_mds=False, metric="precomputed", n_init=red["mds_n_init"],
              max_iter=red["mds_max_iter"], init="random", normalized_stress=True,
              random_state=cfg["random_state"])
    Y = mds.fit_transform(D)
    res = {"stress_1": float(mds.stress_)}
    for k in red["trustworthiness_k"]:
        res[f"trustworthiness_k{k}"] = trustworthiness(Z, Y, n_neighbors=k)
        res[f"trustworthiness_gower_k{k}"] = trustworthiness(D, Y, n_neighbors=k, metric="precomputed")
    return Y, res


def merge_small(counts: pd.Series, min_n: int, label: str) -> pd.Series:
    """kategorien mit n < min_n zusammenfassen, bleibt die sammelzeile zu klein, kommt die nächstkleinere dazu."""
    counts = counts.sort_values(ascending=False)
    merged = list(counts[counts < min_n].index)
    if not merged:
        return counts
    while counts[merged].sum() < min_n and len(merged) < len(counts):
        merged.append(counts.drop(index=merged).idxmin())
    name = label + " (" + ", ".join(map(str, merged)) + ")"
    return pd.concat([counts.drop(index=merged), pd.Series({name: int(counts[merged].sum())})])


def sample_overview(df: pd.DataFrame, cfg: dict, spec: dict[str, str], value_de: dict[str, str]) -> pd.DataFrame:
    """deskriptive übersicht der analysestichprobe, kleine zellen zusammengefasst."""
    min_n = cfg["profiling"]["min_cell_n"]
    n = len(df)
    age = df["age_clean"]
    rows = [{"merkmal": "Stichprobe", "auspraegung": "gesamt", "n": n, "anteil": 1.0},
            {"merkmal": "Alter", "auspraegung": f"Median {age.median():g} (IQR {age.quantile(.25):g}–{age.quantile(.75):g})",
             "n": int(age.notna().sum()), "anteil": float(age.notna().mean())}]
    for col, title in spec.items():
        counts = df[col].map(lambda v: value_de.get(str(v), str(v))).value_counts()
        for cat, c in merge_small(counts, min_n, "zusammengefasst").items():
            rows.append({"merkmal": title, "auspraegung": cat, "n": int(c), "anteil": c / n})
    return pd.DataFrame(rows)
