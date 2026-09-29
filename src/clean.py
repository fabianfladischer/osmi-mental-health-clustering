"""bereinigung: stichprobe, alter, geschlecht, regionen und mehrfachauswahl."""

from __future__ import annotations

import re

import pandas as pd


def filter_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """selbstständige raus, nur personen mit tech-bezug (tech-arbeitgeber oder tech-rolle) behalten."""
    log = {"n_roh": len(df)}
    df = df[df["self_employed"] == 0]
    log["n_nach_selbststaendige"] = len(df)
    log["n_selbststaendige_ausgeschlossen"] = log["n_roh"] - len(df)
    # die tech-rolle wurde nur abgefragt, wenn der arbeitgeber kein tech-unternehmen ist
    tech = (df["tech_company"] == 1) | (df["tech_role"] == 1)
    log["n_tech_unternehmen"] = int((df["tech_company"] == 1).sum())
    log["n_tech_rolle_in_nicht_tech"] = int(((df["tech_company"] == 0) & (df["tech_role"] == 1)).sum())
    log["n_ohne_tech_bezug_ausgeschlossen"] = int((~tech).sum())
    df = df[tech]
    log["n_analyse"] = len(df)
    return df.reset_index(drop=True), log


def clean_age(age: pd.Series, cfg: dict) -> tuple[pd.Series, dict]:
    """unplausibles alter auf nan setzen und durch den median ersetzen."""
    c = cfg["cleaning"]
    implausible = (age < c["age_min"]) | (age > c["age_max"])
    cleaned = age.where(~implausible).astype(float)
    median = float(cleaned.median())
    log = {
        "alter_unplausibel_n": int(implausible.sum()),
        "alter_unplausibel_werte": sorted(age[implausible].astype(int).tolist()),
        "alter_median": median,
    }
    return cleaned.fillna(median), log


def _norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[().,!|;]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_gender(raw: pd.Series, cfg: dict) -> tuple[pd.Series, pd.DataFrame]:
    """freitext-geschlechtsangaben per regelwörterbuch auf wenige kategorien abbilden."""
    r = cfg["gender_rules"]
    lab = r["labels"]

    def rule(value) -> str:
        if pd.isna(value):
            return lab["missing"]
        t = _norm_text(value)
        if t in r["missing_exact"]:
            return lab["missing"]
        if t in r["diverse_exact"]:
            return lab["diverse"]
        if t in r["male_exact"]:
            return lab["male"]
        if t in r["female_exact"]:
            return lab["female"]
        tokens = set(re.split(r"[\s/\-|,]+", t))
        # kurze schlüsselwörter wie "nb" nur als ganzes wort prüfen
        for kw in r["diverse_contains"]:
            if (kw in tokens) if len(kw) <= 3 else (kw in t):
                return lab["diverse"]
        # weiblich vor männlich prüfen, weil "female" auch "male" enthält
        if any(kw in t for kw in r["female_contains"]):
            return lab["female"]
        if any(kw in tokens for kw in r["male_contains"]) or t.startswith("male"):
            return lab["male"]
        return lab["diverse"]

    cleaned = raw.map(rule)
    mapping = (
        pd.DataFrame({"rohwert": raw.fillna("(leer)").astype(str).str.strip(), "kategorie": cleaned})
        .groupby(["rohwert", "kategorie"]).size().reset_index(name="n")
        .sort_values(["kategorie", "n"], ascending=[True, False])
    )
    return cleaned, mapping


def publish_gender_mapping(gm: pd.DataFrame, min_n: int, merge_into: str) -> pd.DataFrame:
    """mapping-tabelle zum veröffentlichen: schreibweisen mit n < min_n je kategorie zusammenfassen.

    ist eine solche sammelzeile selbst kleiner als min_n, wird sie mit der sammelzeile von merge_into vereint.
    """
    small = gm[gm.n < min_n].groupby("kategorie").agg(n=("n", "sum"), varianten=("n", "size"))
    tiny = [c for c in small.index[small.n < min_n] if c != merge_into]
    if tiny:
        group = [merge_into, *tiny]
        small.loc[" oder ".join(group)] = small.loc[group].sum()
        small = small.drop(index=group)
    small = small.reset_index()
    small["rohwert"] = small.varianten.map(lambda v: f"{v} weitere Schreibweisen (je n < {min_n})")
    return pd.concat([gm[gm.n >= min_n], small[["rohwert", "kategorie", "n"]]]).sort_values(
        ["kategorie", "n"], ascending=[True, False])


def group_regions(country: pd.Series, cfg: dict) -> pd.Series:
    """feature grouping: länder zu regionen zusammenfassen."""
    lookup = {c: reg for reg, cs in cfg["regions"].items() if reg != "default" for c in cs}
    return country.map(lambda c: lookup.get(c, cfg["regions"]["default"]))


def split_multiselect(s: pd.Series, sep: str, prefix: str, rename: dict | None = None) -> pd.DataFrame:
    """feature splitting: mehrfachauswahl wie "a|b" in 0/1-spalten aufteilen."""
    lists = s.fillna("").astype(str).map(lambda x: [t.strip() for t in x.split(sep) if t.strip()])
    out = pd.DataFrame(index=s.index)
    for val in lists.explode().dropna().value_counts().index:
        name = (rename or {}).get(val, val)
        out[f"{prefix}{name}"] = lists.map(lambda xs, v=val: int(v in xs))
    return out


DIAGNOSIS_SHORT = {
    "Mood Disorder (Depression, Bipolar Disorder, etc)": "affektiv",
    "Anxiety Disorder (Generalized, Social, Phobia, etc)": "angst",
    "Attention Deficit Hyperactivity Disorder": "adhs",
    "Post-traumatic Stress Disorder": "ptbs",
    "Obsessive-Compulsive Disorder": "zwang",
}

ROLE_SHORT = {
    "Back-end Developer": "backend", "Front-end Developer": "frontend", "DevOps/SysAdmin": "devops",
    "Supervisor/Team Lead": "teamlead", "Other": "andere", "Support": "support",
    "One-person shop": "one_person_shop", "Designer": "design", "Executive Leadership": "fuehrung",
    "Dev Evangelist/Advocate": "evangelist", "Sales": "vertrieb", "HR": "hr",
}


def clean_all(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    """alle bereinigungsschritte, gibt daten, log und zusatztabellen zurück."""
    log, tables = {}, {}
    out = df.copy()
    out["age_clean"], age_log = clean_age(out["age"], cfg)
    log.update(age_log)
    out["gender_clean"], tables["gender_mapping"] = normalize_gender(out["gender"], cfg)
    out["region_work"] = group_regions(out["country_work"], cfg)
    out["region_live"] = group_regions(out["country_live"], cfg)

    sep = cfg["cleaning"]["multiselect_separator"]
    roles = split_multiselect(out["work_position"], sep, "role_", rename=ROLE_SHORT)
    diag = split_multiselect(out["diagnoses_professional"], sep, "diag_", rename=DIAGNOSIS_SHORT)
    # seltene diagnosen in einer spalte "sonstige" zusammenfassen
    diag_main = [c for c in diag.columns if c.removeprefix("diag_") in DIAGNOSIS_SHORT.values()]
    diag_other = diag.drop(columns=diag_main).max(axis=1) if len(diag.columns) > len(diag_main) else 0
    diag = diag[diag_main].assign(diag_sonstige=diag_other)
    out = pd.concat([out, roles, diag], axis=1)
    log["n_rollen_kategorien"] = roles.shape[1]
    log["n_rollen_mehrfachnennung"] = int((roles.sum(axis=1) > 1).sum())
    log["n_diagnose_rohvarianten"] = int(
        out["diagnoses_professional"].dropna().str.split(sep).explode().str.strip().nunique())
    log["n_geschlecht_rohvarianten"] = int(df["gender"].dropna().str.strip().nunique())
    log["geschlecht_verteilung"] = out["gender_clean"].value_counts().to_dict()
    log["region_verteilung"] = out["region_work"].value_counts().to_dict()

    out["treatment_sought"] = out["treatment_sought"].map({1: "Yes", 0: "No"})
    return out, log, tables
