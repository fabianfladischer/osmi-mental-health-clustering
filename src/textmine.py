"""tf-idf der freitextantworten, nur zur beschreibung der cluster."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def top_terms_per_cluster(texts: pd.Series, labels: np.ndarray, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """terme mit der größten tf-idf-differenz zwischen einem cluster und dem rest."""
    t = cfg["textmine"]
    mask = texts.notna() & (texts.astype(str).str.strip().str.len() >= t["min_text_length"])
    docs = texts[mask].astype(str).str.lower()
    lab = np.asarray(labels)[mask.to_numpy()]
    vec = TfidfVectorizer(stop_words="english", ngram_range=tuple(t["ngram_range"]),
                          min_df=t["min_df"], max_df=t["max_df"], sublinear_tf=True)
    M = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())
    # texte, die nach stoppwörtern nichts mehr enthalten (z. b. "see above"), verdünnen sonst die mittelwerte
    has_terms = M.getnnz(axis=1) > 0
    M, lab = M[has_terms], lab[has_terms]
    rows = []
    for c in np.unique(lab):
        # differenz statt mittelwert, sonst stehen allgemeine terme wie "mental health" überall oben
        in_c = np.asarray(M[lab == c].mean(axis=0)).ravel()
        out_c = np.asarray(M[lab != c].mean(axis=0)).ravel()
        diff = in_c - out_c
        for rank, j in enumerate(np.argsort(diff)[::-1][: t["top_terms"]], start=1):
            rows.append({"cluster": int(c) + 1, "rang": rank, "term": terms[j],
                         "tfidf_diff": float(diff[j]), "tfidf_mittel": float(in_c[j])})
    log = {"n_texte": int(has_terms.sum()), "n_texte_ohne_terme": int((~has_terms).sum()),
           "n_terme_vokabular": int(len(terms)),
           "n_texte_je_cluster": {str(int(c) + 1): int((lab == c).sum()) for c in np.unique(lab)}}
    return pd.DataFrame(rows), log
