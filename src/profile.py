"""clusterprofile: größen, mittelwerte, profilvariablen und signifikanztests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    """größe und anteil je cluster."""
    s = pd.Series(labels + 1).value_counts().sort_index()
    return pd.DataFrame({"cluster": s.index, "n": s.values, "anteil": s.values / s.sum()})


def raw_means(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """clustermittel auf der 0-1-skala plus zeile für das gesamtmittel."""
    m = X.groupby(labels + 1).mean()
    m.loc["gesamt"] = X.mean()
    return m


def top_features(means: pd.DataFrame, n: int) -> dict[int, list[str]]:
    """die n merkmale mit der größten absoluten abweichung vom gesamtmittel je cluster (0-1-skala)."""
    dev = means.drop(index="gesamt") - means.loc["gesamt"]
    return {c: dev.loc[c].abs().sort_values(ascending=False).index[:n].tolist() for c in dev.index}


def suppress_cells(ct: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """zellen mit n < min_n sperren, dazu eine zweite zelle, wenn in einer zeile oder spalte nur eine gesperrt ist.

    sonst ließe sich die einzelne gesperrte zelle über 100 % bzw. die zeilensumme zurückrechnen.
    """
    hide = ct < min_n
    changed = True
    while changed:
        changed = False
        for col in hide.columns:
            if hide[col].sum() == 1 and (~hide[col]).any():
                hide.loc[ct[col][~hide[col]].idxmin(), col] = True
                changed = True
        for row in hide.index:
            if hide.loc[row].sum() == 1 and (~hide.loc[row]).any():
                hide.loc[row, ct.loc[row][~hide.loc[row]].idxmin()] = True
                changed = True
    return hide


def categorical_profile(df: pd.DataFrame, labels: np.ndarray, var: str, min_n: int) -> pd.DataFrame:
    """spaltenprozente je cluster, kleine zellen (und je zeile/spalte eine zweite) werden gesperrt."""
    ct = pd.crosstab(df[var], labels + 1)
    pct = ct / ct.sum(axis=0)
    out = pct.map(lambda v: f"{100 * v:.1f}".replace(".", ",") + " %")
    out = out.mask(suppress_cells(ct, min_n), "gesperrt")
    out.insert(0, "variable", var)
    return out.reset_index().rename(columns={var: "auspraegung"})


def cramers_v(ct: pd.DataFrame) -> tuple[float, float, float]:
    """chi²-test und cramérs v für eine kreuztabelle."""
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    return float(chi2), float(p), float(v)


# richtwerte für cramérs v je df* = min(r, c) - 1 nach kim (2017, tab. 2): klein, mittel, groß
CRAMERS_V_THRESHOLDS = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35), 3: (0.06, 0.17, 0.29),
                        4: (0.05, 0.15, 0.25), 5: (0.04, 0.13, 0.22)}


def classify_v(v: float, df_star: int) -> str:
    """einordnung von cramérs v als vernachlässigbar, klein, mittel oder groß."""
    small, medium, large = CRAMERS_V_THRESHOLDS[min(max(df_star, 1), 5)]
    return "groß" if v >= large else "mittel" if v >= medium else "klein" if v >= small else "vernachlässigbar"


def significance_table(df: pd.DataFrame, labels: np.ndarray, cat_vars: list[str], num_vars: list[str],
                       cfg: dict) -> pd.DataFrame:
    """chi² mit cramérs v bzw. kruskal-wallis mit epsilon², p-werte zusätzlich bonferroni-korrigiert."""
    pc = cfg["profiling"]
    rows = []
    for var in cat_vars:
        ct = pd.crosstab(df[var], labels)
        # kategorien entfernen, deren erwartete häufigkeit im kleinsten cluster zu klein für den chi²-test ist
        n_all = ct.to_numpy().sum()
        min_col_share = ct.sum(axis=0).min() / n_all
        dropped = ct.index[ct.sum(axis=1) * min_col_share < pc["chi2_min_expected"]].tolist()
        ct = ct.drop(index=dropped)
        chi2, p, v = cramers_v(ct)
        df_star = min(ct.shape) - 1
        rows.append({"variable": var, "test": "Chi²", "statistik": chi2, "df": (ct.shape[0] - 1) * (ct.shape[1] - 1),
                     "p": p, "effekt": v, "effektmass": "Cramérs V", "df_stern": df_star,
                     "einordnung": classify_v(v, df_star),
                     "min_erwartet": float(np.min(chi2_contingency(ct)[3])),
                     "ausgeschlossene_kategorien": "; ".join(map(str, dropped))})
    n = len(labels)
    for var in num_vars:
        groups = [df.loc[labels == c, var] for c in np.unique(labels)]
        h, p = kruskal(*groups)
        rows.append({"variable": var, "test": "Kruskal-Wallis", "statistik": float(h), "df": len(groups) - 1,
                     "p": float(p), "effekt": float(h / (n - 1)), "effektmass": "Epsilon²"})
    out = pd.DataFrame(rows)
    out["rolle"] = np.where(out.variable.isin(pc["circular_vars"]), "zirkulär – nur deskriptiv",
                            np.where(out.variable.isin(pc["construct_near_vars"]), "konstruktnah", "unabhängige Profilvariable"))
    out["p_bonferroni"] = (out["p"] * len(out)).clip(upper=1.0)
    return out


def share_table(df: pd.DataFrame, labels: np.ndarray, var: str, value) -> pd.Series:
    """anteil einer ausprägung je cluster."""
    return (df[var] == value).groupby(labels + 1).mean()
