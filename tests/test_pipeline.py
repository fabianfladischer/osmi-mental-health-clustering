"""tests für aufbereitung, kodierung und reproduzierbarkeit."""

import numpy as np
import pandas as pd
import pytest

from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score

from src import clean, cluster, features, load, profile, reduce, supplement
from src import select as fsel
from src.utils import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data(cfg):
    raw, _ = load.load_raw(cfg)
    df, flog = clean.filter_sample(raw)
    df, _, _ = clean.clean_all(df, cfg)
    X, idx, _ = features.build_workplace_matrix(df, cfg)
    return raw, df, flog, X, idx


def test_raw_shape(data, cfg):
    raw = data[0]
    assert raw.shape == tuple(cfg["data"]["expected_shape"])


def test_filter_counts(data):
    _, df, flog, _, _ = data
    assert (flog["n_roh"], flog["n_nach_selbststaendige"], flog["n_analyse"]) == (1433, 1146, 1131)
    assert (df["self_employed"] == 0).all()
    assert ((df["tech_company"] == 1) | (df["tech_role"] == 1)).all()


def test_golden_results(data, cfg):
    # kernergebnisse festschreiben, damit änderungen an kodierung, pca oder clustering auffallen
    X = data[3]
    p = reduce.fit_pca(reduce.scale(X, cfg["scaling"]["method"]), list(X.columns),
                       cfg["reduction"]["pca_variance_threshold"], cfg["random_state"])
    assert p["n_components"] == 9
    assert p["cum"][8] == pytest.approx(0.83465, abs=1e-4)
    S = p["scores"][:, :9]
    labels = cluster.relabel_by_size(cluster.fit_model("kmeans", S, cfg["clustering"]["final_k"], cfg)[0])
    assert np.bincount(labels).tolist() == [350, 277, 271, 233]
    assert silhouette_score(S, labels) == pytest.approx(0.1584, abs=1e-3)


def test_no_missing_after_imputation(data):
    _, df, _, X, idx = data
    assert not X.isna().any().any()
    assert not idx.isna().any().any()
    assert not df["age_clean"].isna().any()


def test_age_range(data, cfg):
    age = data[1]["age_clean"]
    assert age.between(cfg["cleaning"]["age_min"], cfg["cleaning"]["age_max"]).all()


def test_ordinal_mapping_complete_and_bounded(data, cfg):
    _, df, _, X, _ = data
    for col, mapping in cfg["ordinal_maps"].items():
        assert set(df[col].dropna().unique()) <= set(mapping), col
    assert X.min().min() >= 0 and X.max().max() <= 1


def test_ordinal_direction(cfg):
    # je item muss die günstige antwort höher kodiert sein als die ungünstige
    good_bad = {
        "mh_benefits": ("Yes", "No"), "care_options_known": ("Yes", "No"), "wellness_discussed": ("Yes", "No"),
        "mh_resources": ("Yes", "No"), "anonymity_protected": ("Yes", "No"),
        "leave_ease": ("Very easy", "Very difficult"),
        "mh_neg_consequence_employer": ("No", "Yes"), "ph_neg_consequence_employer": ("No", "Yes"),
        "comfort_coworkers": ("Yes", "No"), "comfort_supervisor": ("Yes", "No"), "mh_parity": ("Yes", "No"),
        "observed_neg_consequences": ("No", "Yes"), "career_hurt": ("No, it has not", "Yes, it has"),
        "coworkers_view_negative": ("No, they do not", "Yes, they do"),
        "unsupportive_response": ("No", "Yes, I experienced"),
    }
    maps = cfg["ordinal_maps"]
    assert set(good_bad) == set(maps)
    for item, (good, bad) in good_bad.items():
        assert maps[item][good] > maps[item][bad], item
        for neutral in ("I don't know", "I am not sure", "Maybe"):
            if neutral in maps[item]:
                assert maps[item][neutral] == 0.5, (item, neutral)


def test_uncertainty_index_counts_only_explicit_answers(cfg):
    items = cfg["uncertainty_items"]
    df = pd.DataFrame([["I don't know"] * len(items), [None] * len(items), ["Yes"] * len(items)], columns=items)
    df.loc[2, items[1]] = "I am not sure"
    out = features.uncertainty_index(df, cfg)
    assert out.tolist() == pytest.approx([1.0, 0.0, 1 / len(items)])


def test_clean_age_uses_median_of_plausible_values(cfg):
    # median der plausiblen werte ist 30, mit dem ausreißer wäre er 35
    out, log = clean.clean_age(pd.Series([20, 30, 40, 200]), cfg)
    assert out.tolist() == [20, 30, 40, 30]
    assert log["alter_unplausibel_n"] == 1


def test_cramers_v_matches_formula():
    ct = pd.DataFrame([[30, 10, 20], [10, 30, 20]])
    chi2, _, v = profile.cramers_v(ct)
    assert chi2 == pytest.approx(chi2_contingency(ct)[0])
    assert v == pytest.approx(np.sqrt(chi2 / (ct.to_numpy().sum() * 1)))


def test_suppress_cells_secondary():
    ct = pd.DataFrame({1: [50, 3, 40], 2: [60, 30, 12], 3: [45, 25, 8]}, index=["a", "b", "c"])
    hide = profile.suppress_cells(ct, 10)
    assert hide.loc["b", 1] and hide.loc["c", 3]
    # keine zeile und keine spalte darf genau eine gesperrte zelle haben
    assert not (hide.sum(axis=0) == 1).any() and not (hide.sum(axis=1) == 1).any()


def test_unknown_answer_raises(data, cfg):
    df = data[1].copy()
    df.loc[0, "mh_benefits"] = "Unbekannt"
    with pytest.raises(ValueError):
        features.encode_ordinal(df, cfg)


def test_gender_categories(data, cfg):
    allowed = set(cfg["gender_rules"]["labels"].values())
    assert set(data[1]["gender_clean"].unique()) <= allowed
    s = pd.Series(["Male", "female ", "Non-binary", None, "M", "Cis-woman"])
    out, _ = clean.normalize_gender(s, cfg)
    assert out.tolist() == ["maennlich", "weiblich", "divers/andere", "keine Angabe", "maennlich", "weiblich"]


def test_gender_mapping_has_no_small_rows(data, cfg):
    _, df, _, _, _ = data
    _, gm = clean.normalize_gender(df["gender"], cfg)
    pub = clean.publish_gender_mapping(gm, 10, cfg["gender_rules"]["labels"]["diverse"])
    assert pub.n.min() >= 10
    assert pub.n.sum() == len(df)


def test_merge_small_reaches_min_n():
    counts = pd.Series({"a": 50, "b": 20, "c": 4, "d": 3})
    out = supplement.merge_small(counts, 10, "rest")
    assert out.min() >= 10 and out.sum() == counts.sum()
    # c + d = 7 < 10, deshalb kommt b dazu
    assert list(out.index) == ["a", "rest (c, d, b)"]


def test_multiselect_split():
    s = pd.Series(["A|B", "B", None])
    out = clean.split_multiselect(s, "|", "x_")
    assert out.sum().to_dict() == {"x_B": 2, "x_A": 1}


def test_protected_attributes_not_in_clustering(data, cfg):
    X = data[3]
    assert not set(cfg["protected_attributes"]) & set(X.columns)


def test_protected_attributes_not_in_baseline(data, cfg):
    Xb = features.build_baseline_matrix(data[1], cfg)
    for col in cfg["protected_attributes"]:
        assert not any(c == col or c.startswith(col + "=") for c in Xb.columns), col


def test_trans_self_description(cfg):
    s = pd.Series(["Transgender woman", "Male (trans, FtM)", "Transitioned, M2F", "Genderqueer"])
    out, _ = clean.normalize_gender(s, cfg)
    assert out.tolist() == ["weiblich", "maennlich", "weiblich", "divers/andere"]


def test_correlation_filter_threshold(data, cfg):
    X = data[3]
    X_sel, rho, _ = fsel.correlation_filter(X, cfg["selection"]["corr_threshold"])
    sub = rho.loc[X_sel.columns, X_sel.columns].abs().to_numpy().copy()
    np.fill_diagonal(sub, 0)
    assert sub.max() <= cfg["selection"]["corr_threshold"]


def test_determinism(data, cfg):
    X = data[3]
    Z = reduce.scale(X, cfg["scaling"]["method"])
    thr = cfg["reduction"]["pca_variance_threshold"]
    p1 = reduce.fit_pca(Z, list(X.columns), thr, cfg["random_state"])
    p2 = reduce.fit_pca(Z, list(X.columns), thr, cfg["random_state"])
    S = p1["scores"][:, : p1["n_components"]]
    np.testing.assert_allclose(p1["scores"], p2["scores"])
    l1, _ = cluster.fit_model("kmeans", S, cfg["clustering"]["final_k"], cfg)
    l2, _ = cluster.fit_model("kmeans", S, cfg["clustering"]["final_k"], cfg)
    assert (l1 == l2).all()


def test_relabel_by_size():
    labels = np.array([2, 2, 2, 0, 1, 1])
    assert cluster.relabel_by_size(labels).tolist() == [0, 0, 0, 2, 1, 1]
