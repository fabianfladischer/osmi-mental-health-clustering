"""feature-auswahl mit varianzfilter und spearman-korrelationsfilter."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def variance_filter(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.Series]:
    """entfernt fast konstante merkmale und gibt die varianzen mit zurück."""
    vt = VarianceThreshold(threshold=threshold).fit(X)
    variances = pd.Series(vt.variances_, index=X.columns)
    return X.loc[:, vt.get_support()], variances


def correlation_filter(X: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """entfernt nacheinander redundante merkmale mit |rho| über der schwelle."""
    rho = X.corr(method="spearman")
    keep = list(X.columns)
    decisions = []
    while True:
        sub = rho.loc[keep, keep].abs().where(~np.eye(len(keep), dtype=bool))
        max_val = sub.max().max()
        if not max_val > threshold:
            break
        a, b = sub.stack().idxmax()
        # aus dem paar fliegt das merkmal, das insgesamt stärker mit den übrigen korreliert
        mean_corr = sub.mean()
        drop = a if mean_corr[a] >= mean_corr[b] else b
        other = b if drop == a else a
        decisions.append({"entfernt": drop, "partner": other, "rho": float(rho.loc[a, b])})
        keep.remove(drop)
    return X[keep], rho, decisions


def selection_table(candidates: list[str], variances: pd.Series, var_threshold: float,
                    after_var: list[str], corr_decisions: list[dict]) -> pd.DataFrame:
    """tabelle für den anhang: welches merkmal bleibt oder fliegt raus und warum."""
    rows = []
    dropped_corr = {d["entfernt"]: d for d in corr_decisions}
    for c in candidates:
        if c not in after_var:
            reason = f"Varianz {variances[c]:.3f} <= {var_threshold}"
            status = "entfernt"
        elif c in dropped_corr:
            d = dropped_corr[c]
            reason = f"redundant zu {d['partner']} (rho = {d['rho']:.2f})"
            status = "entfernt"
        else:
            reason = "behalten"
            status = "behalten"
        rows.append({"merkmal": c, "varianz": round(float(variances[c]), 4), "status": status, "grund": reason})
    return pd.DataFrame(rows)
