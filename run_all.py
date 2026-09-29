"""komplette analyse von den rohdaten bis zu abbildungen, tabellen und kennzahlen (aufruf: python run_all.py)."""

from __future__ import annotations

import platform
import warnings

import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from src import clean, cluster, features, load, plots, profile, reduce, supplement, textmine
from src import select as fsel
from src.utils import load_config, out_path, save_metrics, set_style, timer

# futurewarnings aus pandas/sklearn ändern nichts am ergebnis und machen die ausgabe nur unübersichtlich
warnings.filterwarnings("ignore", category=FutureWarning)


def save_csv(df: pd.DataFrame, cfg: dict, name: str, index: bool = False) -> None:
    df.to_csv(out_path(cfg, "results", name), index=index, encoding="utf-8")


def pca_space(X: pd.DataFrame, method: str, cfg: dict) -> tuple[np.ndarray, dict]:
    """skalieren, pca rechnen und nur die komponenten bis zur varianzschwelle zurückgeben."""
    Z = reduce.scale(X, method)
    p = reduce.fit_pca(Z, list(X.columns), cfg["reduction"]["pca_variance_threshold"], cfg["random_state"])
    return p["scores"][:, : p["n_components"]], p


def main() -> None:
    cfg = load_config()
    set_style()
    metrics: dict = {}
    rs = cfg["random_state"]
    ccfg = cfg["clustering"]
    k_final = ccfg["final_k"]
    metrics["umgebung"] = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__}

    # --- daten und eda
    with timer(metrics, "A_eda"):
        raw, questions = load.load_raw(cfg)
        metrics["daten"] = {
            "sha256": load.sha256(load.download(cfg)),
            "n_zeilen": raw.shape[0], "n_spalten": raw.shape[1],
            "n_numerisch": int(raw.select_dtypes("number").shape[1]),
            "n_text": int(raw.shape[1] - raw.select_dtypes("number").shape[1]),
            "fehlwertquote_gesamt": float(raw.isna().mean().mean()),
            "n_spalten_mit_fehlwerten": int((raw.isna().mean() > 0).sum()),
            "n_spalten_fehlwerte_ueber_50": int((raw.isna().mean() > 0.5).sum()),
            "fehlwertquote_je_spalte": raw.isna().mean().round(4).to_dict(),
            "arbeitgeberfragen_fehlend_n": int(raw["mh_benefits"].isna().sum()),
            "arbeitgeberfragen_fehlend_alle_selbststaendig": bool(
                (raw.loc[raw["mh_benefits"].isna(), "self_employed"] == 1).all()),
            "selbststaendige_n": int((raw["self_employed"] == 1).sum()),
            "alter_min_roh": int(raw["age"].min()), "alter_max_roh": int(raw["age"].max()),
            "top_laender_arbeit": raw["country_work"].value_counts().head(5).to_dict(),
            "anteil_usa_arbeit": float((raw["country_work"] == "United States of America").mean()),
            "remote_verteilung": raw["remote"].value_counts().to_dict(),
            "unternehmensgroesse_verteilung": raw["company_size"].value_counts().to_dict(),
            "n_freitext_interview_mh_why": int(raw["interview_mh_why"].notna().sum()),
        }
        codebook = pd.DataFrame([{"code": e["code"], "block": e["block"], "frage": questions[e["code"]],
                                  "fehlwertquote": round(float(raw[e["code"]].isna().mean()), 3),
                                  "n_auspraegungen": int(raw[e["code"]].nunique())} for e in cfg["codebook"]])
        save_csv(codebook, cfg, "tab_codebook.csv")
        plots.missing_matrix(raw, cfg)
        plots.missing_rates(raw, cfg)

    # --- aufbereitung, feature engineering und feature-auswahl
    with timer(metrics, "B_aufbereitung"):
        df, metrics["filter"] = clean.filter_sample(raw)
        metrics["filter"]["anteil_usa_arbeit_analyse"] = float((df["country_work"] == "United States of America").mean())
        df, metrics["bereinigung"], tables = clean.clean_all(df, cfg)
        # geschlechts-mapping nur aggregiert speichern, seltene schreibweisen werden zusammengefasst
        gm = tables["gender_mapping"]
        min_n = cfg["profiling"]["min_cell_n"]
        gm_pub = clean.publish_gender_mapping(gm, min_n, cfg["gender_rules"]["labels"]["diverse"])
        save_csv(gm_pub, cfg, "tab_geschlecht_mapping.csv")
        metrics["bereinigung"]["geschlecht_varianten_unter_min_n"] = int((gm.n < min_n).sum())
        plots.age_distribution(df["age"], df["age_clean"], cfg)
        plots.gender_raw(df["gender"], cfg)

        X, idx, metrics["features"] = features.build_workplace_matrix(df, cfg)
        metrics["features"]["fehlend_gesamt_nach_imputation"] = int(X.isna().sum().sum())
        metrics["features"]["anteil_weiss_nicht_je_item"] = (
            df[cfg["uncertainty_items"]].isin(cfg["uncertain_answers"]).mean().round(4).to_dict())
        metrics["features"]["anteil_ungesichert_mindestens_ein_item"] = float((idx["uncertainty_index"] > 0).mean())
        plots.item_distribution(df, X, cfg)

        sel = cfg["selection"]
        X_var, variances = fsel.variance_filter(X, sel["variance_threshold"])
        X_sel, rho, decisions = fsel.correlation_filter(X_var, sel["corr_threshold"])
        tab_sel = fsel.selection_table(list(X.columns), variances, sel["variance_threshold"],
                                       list(X_var.columns), decisions)
        save_csv(tab_sel, cfg, "tab_feature_auswahl.csv")
        abs_rho = rho.abs().where(~np.eye(len(rho), dtype=bool))
        pair = abs_rho.stack().idxmax()
        metrics["auswahl"] = {
            "n_kandidaten": X.shape[1], "n_nach_varianz": X_var.shape[1], "n_final": X_sel.shape[1],
            "entfernt_korrelation": decisions, "max_abs_rho": float(abs_rho.max().max()),
            "max_abs_rho_paar": list(pair), "min_varianz": float(variances.min()),
            "min_varianz_merkmal": variances.idxmin(), "varianzen": variances.round(4).to_dict(),
            "final_merkmale": list(X_sel.columns),
            "n_paare_rho_ueber_berichtsschwelle": int((abs_rho > sel["rho_report_threshold"]).sum().sum() / 2),
            "n_eindeutige_muster": int(X_sel.drop_duplicates().shape[0]),
        }
        plots.correlation_heatmap(rho, cfg)
        # gemeinsamer vergleichsraum: die 16 merkmale auf der 0-1-skala (endmodell ohne pca-kürzung)
        Z = reduce.scale(X_sel, cfg["scaling"]["method"])

    # --- iteration 1: baseline, k-means auf allen übrigen spalten (one-hot)
    with timer(metrics, "D_iteration1_baseline"):
        Xb = features.build_baseline_matrix(df, cfg)
        Xb_var, _ = fsel.variance_filter(Xb, sel["baseline_variance_threshold"])
        Zb = StandardScaler().fit_transform(Xb_var)
        scan_b = cluster.scan_k(Zb, cfg, ["kmeans"], X_ref=Z)
        save_csv(scan_b, cfg, "tab_scan_iteration1_baseline.csv")
        lab_b, _ = cluster.fit_model("kmeans", Zb, k_final, cfg)
        # welche merkmale trennen die baseline-cluster am stärksten
        zb = pd.DataFrame(Zb, columns=Xb_var.columns).groupby(lab_b).mean()
        drivers = (zb.max() - zb.min()).sort_values(ascending=False).head(ccfg["baseline_top_drivers"])
        metrics["iteration1"] = {
            "n_merkmale_onehot": Xb.shape[1], "n_merkmale_nach_varianz": Xb_var.shape[1],
            "silhouette_je_k": dict(zip(scan_b.k, scan_b.silhouette)),
            "davies_bouldin_je_k": dict(zip(scan_b.k, scan_b.davies_bouldin)),
            "bestes_k_silhouette": int(scan_b.loc[scan_b.silhouette.idxmax(), "k"]),
            "silhouette_max": float(scan_b.silhouette.max()),
            "trennende_merkmale_k_final": drivers.round(3).to_dict(),
            "clustergroessen_k_final": np.bincount(lab_b).tolist(),
        }

    # --- iteration 2: arbeitsplatz-merkmale z-standardisiert
    with timer(metrics, "D_iteration2_zstandard"):
        S2, p2 = pca_space(X_sel, cfg["scaling"]["compare_method"], cfg)
        scan_2 = cluster.scan_k(S2, cfg, ["kmeans"], X_ref=Z)
        save_csv(scan_2, cfg, "tab_scan_iteration2_zstandard.csv")
        lab_2, _ = cluster.fit_model("kmeans", S2, k_final, cfg)
        lab_2w, _ = cluster.fit_model("ward", S2, k_final, cfg)
        # prüfen, ob das kleinste cluster nur aus den personen mit beobachteten negativen folgen besteht
        obs = (X_sel["observed_neg_consequences"] == 0).to_numpy() if "observed_neg_consequences" in X_sel else None
        artifact = {}
        if obs is not None:
            for name, lab in (("kmeans", lab_2), ("ward", lab_2w)):
                smallest = np.argmin(np.bincount(lab))
                m = lab == smallest
                artifact[name] = {"n_kleinstes_cluster": int(m.sum()),
                                  "anteil_beobachtete_folgen_im_kleinsten": float(obs[m].mean()),
                                  "anteil_erfasst_von_allen_beobachtenden": float(m[obs].mean())}
        metrics["iteration2"] = {
            "pca_n_komponenten": p2["n_components"], "pca_evr_hk1": float(p2["evr"][0]),
            "silhouette_kmeans_je_k": dict(zip(scan_2.k, scan_2.silhouette)),
            "min_anteil_kmeans_je_k": dict(zip(scan_2.k, scan_2.min_share)),
            "n_beobachtete_neg_folgen": int(obs.sum()) if obs is not None else None,
            "artefakt_cluster": artifact,
            "clustergroessen_kmeans_k_final": np.bincount(lab_2).tolist(),
        }

    # --- dimensionsreduktion im finalen raum (0-1-skala)
    with timer(metrics, "C_reduktion"):
        red = reduce.run_reduction(X_sel, Z, cfg)
        pca = red["pca"]
        S = pca["scores"][:, : pca["n_components"]]
        save_csv(pca["loadings"].round(4), cfg, "tab_pca_ladungen.csv", index=True)
        metrics["reduktion"] = {
            "pca_evr": pca["evr"].tolist(), "pca_kumuliert": pca["cum"].tolist(),
            "pca_n_komponenten": pca["n_components"], "pca_schwelle": cfg["reduction"]["pca_variance_threshold"],
            "pca_kum_bei_n": float(pca["cum"][pca["n_components"] - 1]),
            "pca_eigenwerte": pca["eigenvalues"].round(5).tolist(),
            "pca_zstandard_evr": p2["evr"].tolist(),
            "lle_beste_nachbarn": red["lle_best_neighbors"], "pca_zstandard_n_komponenten": p2["n_components"],
            "pca_top_ladungen": {f"PC{i + 1}": pca["loadings"][f"PC{i + 1}"].abs().sort_values(ascending=False)
                                 .head(4).index.tolist() for i in range(cfg["reduction"]["pca_loadings_components"])},
            "mds_stress": red["mds_stress"], "lle_rekonstruktionsfehler": red["lle_error"],
            "trustworthiness": red["trustworthiness"],
            "laufzeit_sek": red["runtime"],
            "gower_mittel": float(red["gower"][np.triu_indices_from(red["gower"], 1)].mean()),
        }
        dr_tab = pd.DataFrame([{"verfahren": n, **{f"trustworthiness_k{k}": red["trustworthiness"][f"{n}_k{k}"]
                                                     for k in cfg["reduction"]["trustworthiness_k"]},
                                "laufzeit_s": red["runtime"][n]} for n in ("PCA", "MDS", "LLE")])
        save_csv(dr_tab, cfg, "tab_reduktion_vergleich.csv")
        plots.pca_scree(pca, p2["evr"], cfg)
        plots.pca_loadings(pca, cfg)
        plots.mds_lle_quality(red, cfg)

    # --- iteration 3: verfahrensvergleich und endmodell
    with timer(metrics, "D_iteration3_final"):
        scan = cluster.scan_k(S, cfg, ["kmeans", "gmm", "ward"], X_ref=Z)
        save_csv(scan, cfg, "tab_scan_iteration3_final.csv")
        # bic-verlauf der gmm über einen größeren suchbereich, um ein mögliches minimum zu finden
        bic = cluster.gmm_bic_scan(S, cfg)
        save_csv(bic, cfg, "tab_gmm_bic.csv")
        best_gmm = bic.loc[bic.bic.idxmin()]
        # für den verfahrensvergleich bei k_final den kovarianztyp mit dem kleinsten bic bei genau diesem k nehmen
        at_k = bic[bic.k == k_final]
        best_cov = at_k.loc[at_k.bic.idxmin(), "kovarianz"]
        bic_info = cluster.describe_bic(bic)
        # größe der kleinsten komponente im bic-optimum, um entartete lösungen zu erkennen
        k_opt = int(best_gmm.k)
        lab_opt, _ = cluster.fit_model("gmm", S, k_opt, cfg, best_gmm.kovarianz)
        final_row = scan[(scan.verfahren == ccfg["final_method"]) & (scan.k == k_final)].iloc[0]
        km = scan[scan.verfahren == "kmeans"]
        metrics["iteration3"] = {
            "scan": scan.round(4).to_dict(orient="records"),
            "gmm_bic_optimum": {"kovarianz": best_gmm.kovarianz, "k": k_opt, "bic": float(best_gmm.bic),
                                "kleinste_komponente_n": int(np.bincount(lab_opt, minlength=k_opt).min())},
            "gmm_bic_suchbereich": ccfg["gmm_bic_k_range"],
            "gmm_bic_verlauf": bic_info,
            "gmm_bic_bestes_k_je_typ": {c: v["bestes_k"] for c, v in bic_info.items()},
            "gmm_kovarianz_vergleich": best_cov,
            "kmeans_bestes_k_silhouette": int(km.loc[km.silhouette.idxmax(), "k"]),
            "kmeans_silhouette_max": float(km.silhouette.max()),
            "final": {"verfahren": ccfg["final_method"], "k": k_final, **final_row.drop(["verfahren", "k"]).to_dict()},
        }

        labels_raw, _ = cluster.fit_model(ccfg["final_method"], S, k_final, cfg)
        labels = cluster.relabel_by_size(labels_raw)

        # bootstrap: k-means für alle k, gmm und ward nur beim finalen k
        rows = []
        k_lo, k_hi = ccfg["k_range"]
        for k in range(k_lo, k_hi + 1):
            for a in cluster.bootstrap_stability(S, "kmeans", k, cfg):
                rows.append({"verfahren": "kmeans", "k": k, "ari": a})
        for meth, cov in (("gmm", best_cov), ("ward", "full")):
            for a in cluster.bootstrap_stability(S, meth, k_final, cfg, cov):
                rows.append({"verfahren": meth, "k": k_final, "ari": a})
        boot = pd.DataFrame(rows)
        save_csv(boot.groupby(["verfahren", "k"]).ari.describe().reset_index(), cfg, "tab_bootstrap_stabilitaet.csv")
        q = ccfg["bootstrap_quantile"]
        bstat = boot.groupby(["verfahren", "k"]).ari.agg(["mean", "std", lambda s: s.quantile(q), "min"])
        bstat.columns = ["mittel", "std", "q05", "min"]

        lab_gmm, gmm_model = cluster.fit_model("gmm", S, k_final, cfg, best_cov)
        lab_ward, _ = cluster.fit_model("ward", S, k_final, cfg)
        lab_full, _ = cluster.fit_model("kmeans", Z, k_final, cfg)
        idx_mat = idx[list(cfg["indices"]) + ["uncertainty_index"]]
        lab_idx, _ = cluster.fit_model("kmeans", idx_mat.to_numpy(), k_final, cfg)
        label_sets = {"k-Means": labels, f"GMM ({best_cov})": lab_gmm, "Ward": lab_ward,
                      "k-Means ohne PCA": lab_full, "Index-Variante": lab_idx,
                      "Iteration 2 (z)": lab_2, "Baseline": lab_b}
        ari = cluster.ari_matrix(label_sets)
        save_csv(ari.round(4), cfg, "tab_ari_matrix.csv", index=True)
        # silhouetten aus verschiedenen räumen sind nicht vergleichbar, deshalb alle lösungen zusätzlich in z bewerten
        metrics["vergleichsraum"] = {
            "silhouette_k_final": {n: silhouette_score(Z, lab) for n, lab in label_sets.items()},
            "davies_bouldin_k_final": {n: davies_bouldin_score(Z, lab) for n, lab in label_sets.items()},
            "silhouette_je_k_baseline": dict(zip(scan_b.k, scan_b.silhouette_ref)),
            "silhouette_je_k_iteration2": dict(zip(scan_2.k, scan_2.silhouette_ref)),
            "silhouette_je_k_final": dict(zip(km.k, km.silhouette_ref)),
        }

        proba = gmm_model.predict_proba(S).max(axis=1)
        thr = ccfg["gmm_posterior_threshold"]

        # endmodell direkt mit der alternative k = compare_k vergleichen
        k_cmp = ccfg["compare_k"]
        lab_cmp = cluster.relabel_by_size(cluster.fit_model("kmeans", S, k_cmp, cfg)[0])
        cross = pd.crosstab(pd.Series(lab_cmp + 1, name=f"k{k_cmp}"), pd.Series(labels + 1, name=f"k{k_final}"))
        save_csv(cross, cfg, f"tab_vergleich_k{k_cmp}_k{k_final}.csv", index=True)
        means_cmp = profile.raw_means(X_sel, lab_cmp)
        save_csv(means_cmp.round(4), cfg, f"tab_cluster_mittelwerte_k{k_cmp}.csv", index=True)
        metrics["stabilitaet"] = {
            "bootstrap_resamples": ccfg["bootstrap_resamples"],
            "kmeans_ari_mittel_je_k": bstat.loc["kmeans", "mittel"].to_dict(),
            "kmeans_ari_q05_je_k": bstat.loc["kmeans", "q05"].to_dict(),
            "final_ari_mittel": float(bstat.loc[("kmeans", k_final), "mittel"]),
            "final_ari_std": float(bstat.loc[("kmeans", k_final), "std"]),
            "final_ari_q05": float(bstat.loc[("kmeans", k_final), "q05"]),
            "final_ari_min": float(bstat.loc[("kmeans", k_final), "min"]),
            "final_min_share_ok": bool(np.bincount(labels).min() / len(labels) >= ccfg["min_cluster_share"]),
            "vergleich_k": k_cmp,
            "vergleich_groessen": {i + 1: int(n) for i, n in enumerate(np.bincount(lab_cmp))},
            "vergleich_kreuztabelle": {int(r): {int(c): int(v) for c, v in row.items()} for r, row in
                                       cross.to_dict(orient="index").items()},
            "vergleich_ari": float(adjusted_rand_score(lab_cmp, labels)),
            "vergleich_mittelwerte": means_cmp.round(4).to_dict(orient="index"),
            "gmm_ari_mittel": float(bstat.loc[("gmm", k_final), "mittel"]),
            "ward_ari_mittel": float(bstat.loc[("ward", k_final), "mittel"]),
            "ari_matrix": ari.round(4).to_dict(),
            "gmm_posterior_schwelle": thr,
            "gmm_anteil_max_posterior_unter_schwelle": float((proba < thr).mean()),
            "gmm_max_posterior_median": float(np.median(proba)),
            "index_variante_silhouette": float(silhouette_score(idx_mat, lab_idx)),
        }

        Zlink = linkage(S, method="ward")
        # fusionshöhen der letzten schritte (index 0 = übergang von 1 auf 2 cluster) für die dendrogramm-aussage
        metrics["iteration3"]["ward_fusionshoehen"] = Zlink[::-1, 2][:8].tolist()
        plots.baseline_elbow(scan_b, scan, cfg)
        plots.model_comparison(scan, bic, best_cov, cfg)
        plots.ward_dendrogram(Zlink, k_final, cfg)
        plots.silhouette_plot(S, labels, cfg)
        plots.stability(boot, ari, cfg)
        plots.embeddings_by_cluster(red["embeddings_2d"], labels, red["trustworthiness"], cfg, pca["evr"])

    # --- clusterprofile
    with timer(metrics, "E_profile"):
        pc = cfg["profiling"]
        sizes = profile.cluster_sizes(labels)
        save_csv(sizes, cfg, "tab_clustergroessen.csv")
        prof_X = pd.concat([X_sel, idx.drop(columns=[c for c in idx.columns if c in X_sel.columns])], axis=1)
        means = profile.raw_means(prof_X, labels)
        save_csv(means.round(4), cfg, "tab_cluster_mittelwerte.csv", index=True)
        # für die grafiken den unsicherheitsindex umdrehen, damit überall höher = günstiger gilt
        X_disp = X_sel.rename(columns={"uncertainty_index": "informedness"})
        X_disp["informedness"] = 1 - X_disp["informedness"]
        means_items = profile.raw_means(X_disp, labels)
        top = profile.top_features(means_items, pc["top_features_per_cluster"])

        profiles = pd.concat([profile.categorical_profile(df, labels, v, pc["min_cell_n"])
                              for v in pc["categorical_vars"]])
        save_csv(profiles, cfg, "tab_profilvariablen.csv")
        desc = df.assign(cluster=labels + 1, **idx).groupby("cluster")[pc["numeric_vars"]].agg(
            ["mean", "median", "std"]).round(3)
        save_csv(desc, cfg, "tab_deskriptiv_numerisch.csv", index=True)
        sig = profile.significance_table(df.assign(**idx), labels, pc["categorical_vars"], pc["numeric_vars"], cfg)
        save_csv(sig, cfg, "tab_signifikanz.csv")

        spec = {  # beschriftung -> (variable, wert)
            "Aktuell psych. Erkrankung (ja)": ("mh_current", "Yes"),
            "Früher psych. Erkrankung (ja)": ("mh_past", "Yes"),
            "Ärztliche Diagnose (ja)": ("diagnosed_professional", "Yes"),
            "Je in Behandlung (ja)": ("treatment_sought", "Yes"),
            "Unbehandelt oft beeinträchtigt": ("interference_untreated", "Often"),
            "Thema im Bewerbungsgespräch: nein": ("interview_mh", "No"),
            "Unternehmen > 1000 Beschäftigte": ("company_size", "More than 1000"),
            "Unternehmen 26–100 Beschäftigte": ("company_size", "26-100"),
            "Immer remote": ("remote", "Always"),
            "Diagnose affektive Störung": ("diag_affektiv", 1),
            "Diagnose Angststörung": ("diag_angst", 1),
            "Diagnose ADHS": ("diag_adhs", 1),
            "Rolle Führung/Teamleitung": ("role_teamlead", 1),
            "Geschlecht weiblich": ("gender_clean", "weiblich"),
            "Arbeitsort Nordamerika": ("region_work", "Nordamerika"),
        }
        shares = pd.DataFrame({lbl: profile.share_table(df, labels, v, val) for lbl, (v, val) in spec.items()}).T
        counts = pd.DataFrame({lbl: (df[v] == val).groupby(labels + 1).sum() for lbl, (v, val) in spec.items()}).T
        save_csv(shares.round(4), cfg, "tab_profil_anteile.csv", index=True)
        overall = {lbl: float((df[v] == val).mean()) for lbl, (v, val) in spec.items()}

        terms, tlog = textmine.top_terms_per_cluster(df[cfg["textmine"]["column"]], labels, cfg)
        save_csv(terms.round(5), cfg, "tab_tfidf_terme.csv")

        metrics["profile"] = {
            "groessen": sizes.set_index("cluster").n.to_dict(),
            "anteile": sizes.set_index("cluster").anteil.round(4).to_dict(),
            "mittelwerte": means.round(4).to_dict(orient="index"),
            "top_merkmale": top,
            "profil_anteile": shares.where(counts >= pc["min_cell_n"]).round(4).to_dict(),
            "profil_anteile_gesamt": overall,
            "profil_zellen_unterdrueckt": int((counts < pc["min_cell_n"]).sum().sum()),
            "alter_median_je_cluster": df.groupby(labels + 1)["age_clean"].median().to_dict(),
            "signifikanz": sig.to_dict(orient="records"),
            "textmining": tlog,
            "tfidf_top_terme": terms.groupby("cluster").term.apply(list).to_dict(),
        }
        plots.cluster_heatmap(means_items, sizes, cfg)
        plots.cluster_profiles(means_items, top, cfg)
        plots.profile_variables(shares, counts, cfg)
        plots.tfidf_terms(terms, cfg)
        plots.index_space(idx, labels, cfg)

        # alle übrigen umfragemerkmale (baseline-matrix) in 2d, eingefärbt nach finalem cluster
        Yb = PCA(n_components=2, random_state=rs).fit_transform(Zb)
        tw_b = trustworthiness(Zb, Yb, n_neighbors=cfg["reduction"]["trust_select_k"])
        metrics["reduktion"]["gesamtdaten_pca_trustworthiness"] = tw_b
        metrics["reduktion"]["gesamtdaten_n_merkmale"] = Zb.shape[1]
        metrics["reduktion"]["gesamtdaten_ari_final_vs_baseline_k4"] = float(adjusted_rand_score(labels, lab_b))
        plots.baseline_space(Yb, labels, tw_b, cfg)

    # --- ergänzende analysen: k = 2, kreuztabellen, calinski-harabasz, sensitivität, nicht-metrische mds
    with timer(metrics, "F_ergaenzungen"):
        lab_de = cfg["labels_de"]
        # k = 2 als einfachste lösung neben k = 3 und k = 4
        lab_k2 = cluster.relabel_by_size(cluster.fit_model("kmeans", S, 2, cfg)[0])
        b2 = boot[(boot.verfahren == "kmeans") & (boot.k == 2)].ari
        means_k2 = profile.raw_means(X_sel, lab_k2)
        save_csv(means_k2.round(4), cfg, "tab_cluster_mittelwerte_k2.csv", index=True)
        cross_k2 = supplement.crosstab_with_sums(lab_k2, labels, "k2", f"k{k_final}")
        save_csv(cross_k2, cfg, f"tab_anhang_kreuztabelle_k2_k{k_final}.csv", index=True)
        cross_cmp = supplement.crosstab_with_sums(lab_cmp, labels, f"k{k_cmp}", f"k{k_final}")
        save_csv(cross_cmp, cfg, f"tab_anhang_kreuztabelle_k{k_cmp}_k{k_final}.csv", index=True)
        # trennachse: anteil des quadrierten abstands der beiden zentroide je hauptkomponente
        cen = np.vstack([S[lab_k2 == c].mean(axis=0) for c in (0, 1)])
        d2 = (cen[0] - cen[1]) ** 2
        diff_k2 = (means_k2.loc[1] - means_k2.loc[2]).sort_values(key=abs, ascending=False)
        k2 = {
            **cluster.internal_metrics(S, lab_k2),
            "silhouette_gemeinsamer_raum": silhouette_score(Z, lab_k2),
            "bootstrap_ari_mittel": float(b2.mean()), "bootstrap_ari_median": float(b2.median()),
            "bootstrap_ari_q05": float(b2.quantile(ccfg["bootstrap_quantile"])),
            "groessen": {i + 1: int(n) for i, n in enumerate(np.bincount(lab_k2))},
            "mittelwerte": means_k2.round(4).to_dict(orient="index"),
            "differenz_c1_minus_c2": diff_k2.round(4).to_dict(),
            "index_mittelwerte": idx.groupby(lab_k2 + 1).mean().round(4).to_dict(orient="index"),
            "zentroidabstand_anteil_je_hk": dict(zip([f"HK{i + 1}" for i in range(len(d2))], (d2 / d2.sum()).round(4))),
            "hk1_mittel_je_cluster": {i + 1: float(S[lab_k2 == i, 0].mean()) for i in (0, 1)},
            "kreuztabelle_k4": {str(r): {str(c): int(v) for c, v in row.items()} for r, row in
                                cross_k2.to_dict(orient="index").items()},
            "ari_zu_k4": adjusted_rand_score(lab_k2, labels),
        }

        # calinski-harabasz der iterationen bei k_final: gemeinsamer raum wie in der iterationstabelle und eigener raum
        ch_iter = {name: {"gemeinsamer_raum": calinski_harabasz_score(Z, lab),
                          "eigener_raum": calinski_harabasz_score(space, lab)}
                   for name, lab, space in (("Baseline", lab_b, Zb), ("Iteration 2 (z)", lab_2, S2),
                                            ("Iteration 3 (final)", labels, S))}
        ch_k = {int(k): {"modellraum": float(km.loc[km.k == k, "calinski_harabasz"].iloc[0]),
                         "gemeinsamer_raum": calinski_harabasz_score(Z, cluster.fit_model("kmeans", S, int(k), cfg)[0])}
                for k in range(2, 7)}

        # je clustering-item: fehlende (nicht zutreffende) antworten mit 0,5, weiß-nicht-antworten mit 0,5
        items = cfg["clustering_items"]
        besetzung = {c: {"fehlend_mit_0_5": int(df[c].isna().sum()),
                         "weiss_nicht_mit_0_5": int(df[c].isin(cfg["uncertain_answers"]).sum()),
                         "wert_0_5_gesamt": int((X_sel[c] == 0.5).sum())} for c in items}
        besetzung["mh_benefits"]["nicht_berechtigt_mit_0"] = int(
            (df["mh_benefits"] == "Not eligible for coverage / N/A").sum())

        indizes = {name: {"items": spec["items"], "items_de": [lab_de[i] for i in spec["items"]],
                          "invertiert": spec["invert"], "n_items": len(spec["items"])}
                   for name, spec in cfg["indices"].items()}
        indizes["uncertainty_index"] = {"items": cfg["uncertainty_items"],
                                        "items_de": [lab_de[i] for i in cfg["uncertainty_items"]],
                                        "invertiert": False, "n_items": len(cfg["uncertainty_items"]),
                                        "gezaehlte_antworten": cfg["uncertain_answers"]}

        gm_pub = pd.read_csv(out_path(cfg, "results", "tab_geschlecht_mapping.csv"))
        geschlecht = {"rohvarianten_ohne_fehlend": metrics["bereinigung"]["n_geschlecht_rohvarianten"],
                      "fehlend_n": int(df["gender"].isna().sum()),
                      "rohvarianten_inkl_fehlend": int(df["gender"].fillna("(leer)").astype(str).str.strip().nunique()),
                      "keine_angabe_n": int((df["gender_clean"] == cfg["gender_rules"]["labels"]["missing"]).sum()),
                      "mapping_zeilen": gm_pub.to_dict(orient="records")}

        sens = supplement.sensitivity(df, X_sel, labels, cfg)
        _, nmds = supplement.nonmetric_mds(red["gower"], Z, cfg)

        spec_overview = {"gender_clean": "Geschlecht", "region_work": "Arbeitsregion",
                         "company_size": "Unternehmensgröße (Beschäftigte)", "remote": "Remote-Arbeit",
                         "tech_company": "Tech-Unternehmen", "mh_current": "Aktuell psych. Erkrankung",
                         "diagnosed_professional": "Ärztliche Diagnose", "treatment_sought": "Je in Behandlung"}
        value_de = {"maennlich": "männlich", "Yes": "ja", "No": "nein", "Maybe": "vielleicht", "1.0": "ja",
                    "0.0": "nein", "Sometimes": "manchmal", "Never": "nie", "Always": "immer",
                    "More than 1000": "mehr als 1000"}
        overview = supplement.sample_overview(df, cfg, spec_overview, value_de)
        save_csv(overview.round(4), cfg, "tab_anhang_stichprobe.csv")

        metrics["ergaenzungen"] = {
            "k2": k2,
            "n_init": {"kmeans": ccfg["kmeans_n_init"], "gmm": ccfg["gmm_n_init"],
                       "mds": cfg["reduction"]["mds_n_init"]},
            "kreuztabelle_k3_k4": {str(r): {str(c): int(v) for c, v in row.items()} for r, row in
                                   cross_cmp.to_dict(orient="index").items()},
            "calinski_harabasz_iterationen_k4": ch_iter,
            "calinski_harabasz_kmeans_je_k": ch_k,
            "besetzung_0_5_je_item": besetzung,
            "indizes": indizes,
            "bonferroni_tests": {"n_tests": len(sig), "n_chi2": int((sig.test == "Chi²").sum()),
                                 "n_kruskal_wallis": int((sig.test != "Chi²").sum()),
                                 "je_rolle": sig.rolle.value_counts().to_dict()},
            "geschlecht": geschlecht,
            "anteil_mind_ein_weiss_nicht": metrics["features"]["anteil_ungesichert_mindestens_ein_item"],
            "anteil_nordamerika_je_cluster": shares.loc["Arbeitsort Nordamerika"].round(4).to_dict(),
            "sensitivitaet": sens,
            "mds_nichtmetrisch": {**nmds, "metrisch_stress_1": red["mds_stress"][2],
                                  **{f"metrisch_trustworthiness_k{k}": red["trustworthiness"][f"MDS_k{k}"]
                                     for k in cfg["reduction"]["trustworthiness_k"]}},
            "stichprobe": overview.round(4).to_dict(orient="records"),
        }

    metrics["laufzeit"]["gesamt"] = sum(metrics["laufzeit"].values())
    save_metrics(metrics, out_path(cfg, "results", "metrics.json"))
    print("Fertig: Ergebnisse in figures/ und results/")


if __name__ == "__main__":
    main()
