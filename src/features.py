"""feature engineering: ordinale kodierung, indizes und one-hot-matrix für die baseline."""

from __future__ import annotations

import pandas as pd

# spalten, die in der baseline nicht sinnvoll sind (freitext, rohtext der mehrfachauswahl, konstante).
# geschützte merkmale kommen zusätzlich aus protected_attributes in der config.
BASELINE_EXCLUDE = [
    "interview_ph_why", "interview_mh_why", "diagnoses_current_yes", "diagnoses_believed_maybe",
    "diagnoses_professional", "work_position", "self_employed",
]


def encode_ordinal(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """arbeitsplatz-items ordinal auf 0 bis 1 kodieren, höher = günstiger."""
    maps = cfg["ordinal_maps"]
    fill = cfg["ordinal_missing_value"]
    X = pd.DataFrame(index=df.index)
    log = {"fehlend_je_item": {}}
    for col, mapping in maps.items():
        # antworten ohne mapping sollen auffallen und nicht still verloren gehen
        unknown = set(df[col].dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unbekannte Antworten in {col}: {unknown}")
        X[col] = df[col].map(mapping)
        n_missing = int(X[col].isna().sum())
        if n_missing:
            log["fehlend_je_item"][col] = n_missing
        X[col] = X[col].fillna(fill)
    return X, log


def uncertainty_index(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """anteil der ausdrücklichen weiß-nicht-antworten über die uncertainty_items."""
    # fehlende antworten zählen nicht: sie kommen fast nur bei personen ohne leistungen vor,
    # dort ist die frage nach den optionen nicht anwendbar und kein wissensmangel
    items = cfg["uncertainty_items"]
    return df[items].isin(cfg["uncertain_answers"]).mean(axis=1).rename("uncertainty_index")


def build_indices(X_ord: pd.DataFrame, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """indizes als mittelwert der zugehörigen items (0 bis 1), stigma wird umgedreht."""
    out = pd.DataFrame(index=X_ord.index)
    for name, spec in cfg["indices"].items():
        s = X_ord[spec["items"]].mean(axis=1)
        out[name] = 1 - s if spec["invert"] else s
    out["uncertainty_index"] = uncertainty_index(df, cfg)
    return out


def build_workplace_matrix(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """merkmale fürs clustering (arbeitsplatz-items und unsicherheitsindex) plus alle indizes."""
    X_ord, log = encode_ordinal(df, cfg)
    idx = build_indices(X_ord, df, cfg)
    X = pd.concat([X_ord[cfg["clustering_items"]], idx[cfg["clustering_generated"]]], axis=1)
    for col in cfg["protected_attributes"]:
        assert col not in X.columns, "Geschützte Merkmale dürfen kein Clustering-Input sein"
    log["n_kandidaten"] = X.shape[1]
    log["index_mittelwerte"] = idx.mean().to_dict()
    return X, idx, log


def build_baseline_matrix(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """baseline: übrige umfragespalten one-hot (fehlend als eigene kategorie) plus rollen, ohne geschützte merkmale."""
    drop = set(BASELINE_EXCLUDE) | set(cfg["protected_attributes"])
    base = df.drop(columns=[c for c in drop if c in df.columns])
    base = base[[c for c in base.columns if not c.startswith(("role_", "diag_"))]]
    cat = base.astype("string").fillna("NA")
    onehot = pd.get_dummies(cat, prefix_sep="=", dtype=float)
    roles = df[[c for c in df.columns if c.startswith("role_")]].astype(float)
    X = pd.concat([onehot, roles], axis=1)
    for col in cfg["protected_attributes"]:
        assert not any(c == col or c.startswith(col + "=") for c in X.columns), col
    return X
