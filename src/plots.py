"""alle abbildungen mit deutscher beschriftung, dezimalkomma und farbenblind-tauglicher palette."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_samples

from .utils import (DIV_CMAP, NEG_COLOR, PALETTE, SEQ_CMAP, de_formatter, de_num, pct_formatter, save_fig)

BLOCK_COLORS = {
    "Arbeitskontext": PALETTE[0], "Arbeitgeber-Angebote": PALETTE[1], "Offenheit/Stigma": PALETTE[2],
    "Selbststaendige": PALETTE[3], "Fruehere Arbeitgeber": PALETTE[4], "Einstellungen/Stigma allgemein": PALETTE[5],
    "Freitext": "#999999", "Eigene Gesundheit": PALETTE[6], "Demografie": "#882255",
}
BLOCK_DE = {"Selbststaendige": "Selbstständige", "Fruehere Arbeitgeber": "Frühere Arbeitgeber"}

# zusätzlich zur farbe unterschiedliche marker je cluster
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _lab(cfg, c):
    return cfg["labels_de"].get(c, c)


# --- eda

def missing_matrix(df_raw: pd.DataFrame, cfg: dict):
    """fehlwert-matrix, spalten nach themenblock und zeilen nach anzahl fehlwerte sortiert."""
    blocks = {e["code"]: e["block"] for e in cfg["codebook"]}
    order = sorted(df_raw.columns, key=lambda c: list(BLOCK_COLORS).index(blocks[c]))
    M = df_raw[order].isna()
    M = M.loc[M.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.imshow(M.to_numpy(), aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#f2f2f2", "#333333"]),
              interpolation="nearest")
    ax.set_xlabel("Merkmale (nach Themenblock gruppiert)")
    ax.set_ylabel("Teilnehmende (sortiert nach Anzahl Fehlwerte)")
    ax.set_yticks([0, len(M) - 1], ["1", de_num(len(M), 0)])
    ax.set_xticks([])
    ax.grid(False)
    # farbige blockleiste unter der matrix
    pos = 0
    for b in BLOCK_COLORS:
        n = sum(blocks[c] == b for c in order)
        if n:
            ax.add_patch(plt.Rectangle((pos - 0.5, len(M) + 8), n, 25, color=BLOCK_COLORS[b], clip_on=False))
            pos += n
    ax.set_ylim(len(M) + 35, -0.5)
    handles = [Patch(color=BLOCK_COLORS[b], label=BLOCK_DE.get(b, b)) for b in BLOCK_COLORS]
    handles += [Patch(color="#333333", label="fehlend"), Patch(color="#f2f2f2", label="vorhanden")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    return save_fig(fig, cfg, "abb_01_fehlwerte_matrix.png")


def missing_rates(df_raw: pd.DataFrame, cfg: dict):
    """fehlwertquote je spalte, eingefärbt nach themenblock."""
    blocks = {e["code"]: e["block"] for e in cfg["codebook"]}
    rates = df_raw.isna().mean()
    rates = rates[rates > 0].sort_values()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(range(len(rates)), rates.values, color=[BLOCK_COLORS[blocks[c]] for c in rates.index], height=0.75)
    ax.set_yticks(range(len(rates)), rates.index, fontsize=6.5)
    ax.set_ylabel("Merkmal (Kurzcode, siehe Codebook)")
    ax.xaxis.set_major_formatter(pct_formatter())
    ax.set_xlabel("Anteil fehlender Werte (n = " + de_num(len(df_raw), 0) + ")")
    ax.grid(axis="y", visible=False)
    used = [b for b in BLOCK_COLORS if b in {blocks[c] for c in rates.index}]
    ax.legend(handles=[Patch(color=BLOCK_COLORS[b], label=BLOCK_DE.get(b, b)) for b in used],
              loc="lower right", fontsize=7)
    return save_fig(fig, cfg, "abb_02_fehlwertquote.png")


def age_distribution(age_raw: pd.Series, age_clean: pd.Series, cfg: dict):
    """altersverteilung vor und nach der bereinigung."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    axes[0].hist(age_raw, bins=60, color=PALETTE[0])
    axes[0].set_title("Rohdaten (Analysestichprobe)")
    out = age_raw[(age_raw < cfg["cleaning"]["age_min"]) | (age_raw > cfg["cleaning"]["age_max"])]
    for n, v in enumerate(sorted(out)):
        axes[0].annotate(de_num(v, 0), (v, 0), xytext=(0, 14 + 12 * (n % 2)), textcoords="offset points", ha="center",
                         fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6))
    axes[1].hist(age_clean, bins=range(15, 80, 2), color=PALETTE[2])
    axes[1].set_title("Bereinigt (unplausible Werte → Median)")
    for ax in axes:
        ax.set_xlabel("Alter in Jahren")
        ax.set_ylabel("Anzahl")
        ax.xaxis.set_major_formatter(de_formatter())
    return save_fig(fig, cfg, "abb_03_alter.png")


def gender_raw(raw: pd.Series, cfg: dict):
    """häufigste rohangaben zum geschlecht, varianten mit n < min_cell_n nur zusammengefasst."""
    min_n = cfg["profiling"]["min_cell_n"]
    counts = raw.fillna("(leer)").astype(str).str.strip().value_counts()
    shown = counts[counts >= min_n]
    rest = counts.drop(shown.index)
    data = pd.concat([shown, pd.Series({f"{de_num(len(rest), 0)} weitere Varianten (je n < {min_n})": rest.sum()})])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.barh(range(len(data)), data.values, color=PALETTE[0], height=0.7)
    ax.set_yticks(range(len(data)), [f"„{s}“" if i < len(shown) else s for i, s in enumerate(data.index)], fontsize=7.5)
    ax.invert_yaxis()
    for i, v in enumerate(data.values):
        ax.text(v + 5, i, de_num(v, 0), va="center", fontsize=7)
    ax.set_xlabel("Anzahl Nennungen")
    ax.grid(axis="y", visible=False)
    return save_fig(fig, cfg, "abb_04_geschlecht_roh.png")


def item_distribution(df: pd.DataFrame, X: pd.DataFrame, cfg: dict):
    """antwortverteilung der arbeitsplatz-items: ungünstig, neutral, weiß nicht, günstig."""
    items = cfg["clustering_items"]
    unsure = df[items].isin(cfg["uncertain_answers"]) | df[items].isna()
    rows = []
    for c in items:
        v = X[c]
        rows.append({"item": _lab(cfg, c),
                     "ungünstig": (v < 0.5).mean(),
                     "neutral/vielleicht": ((v == 0.5) & ~unsure[c]).mean(),
                     "weiß nicht/fehlend": unsure[c].mean(),
                     "günstig": (v > 0.5).mean()})
    t = pd.DataFrame(rows).set_index("item").sort_values("günstig")
    colors = [NEG_COLOR, "#bdbdbd", "#8c8c8c", PALETTE[0]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    left = np.zeros(len(t))
    for col, color in zip(t.columns, colors):
        ax.barh(t.index, t[col], left=left, color=color, label=col, edgecolor="white", linewidth=1, height=0.72)
        for i, (l, w) in enumerate(zip(left, t[col])):
            if w >= 0.08:
                ax.text(l + w / 2, i, de_num(100 * w, 0), ha="center", va="center", fontsize=6.5,
                        color="white" if color in (PALETTE[0], NEG_COLOR, "#8c8c8c") else "black")
        left += t[col].to_numpy()
    ax.xaxis.set_major_formatter(pct_formatter())
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=7.5)
    ax.set_xlabel("Anteil der Teilnehmenden (Werte in %)")
    return save_fig(fig, cfg, "abb_05_arbeitsplatz_items.png")


# --- feature-auswahl

def correlation_heatmap(rho: pd.DataFrame, cfg: dict):
    """spearman-korrelationsmatrix der clustering-merkmale."""
    labels = [_lab(cfg, c) for c in rho.columns]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(rho.to_numpy(), cmap=DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                v = rho.iat[i, j]
                txt = de_num(round(v, 2) + 0.0, 2).replace("-0,00", "0,00")
                ax.text(j, i, txt, ha="center", va="center", fontsize=5.2,
                        color="white" if abs(v) > 0.45 else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, format=de_formatter())
    cb.set_label("Spearman-Rangkorrelation ρ")
    return save_fig(fig, cfg, "abb_06_korrelation.png")


# --- dimensionsreduktion

def pca_scree(pca: dict, pca_cmp_evr: np.ndarray, cfg: dict):
    """erklärte varianz je hauptkomponente und kumuliert."""
    evr, cum, n = pca["evr"], pca["cum"], pca["n_components"]
    x = np.arange(1, len(evr) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    axes[0].bar(x, evr, color=PALETTE[0], width=0.7, label="erklärte Varianz")
    axes[0].plot(x, pca_cmp_evr, "o-", color=PALETTE[1], ms=4, lw=1.5, label="Vergleich: z-standardisiert")
    axes[0].set_xlabel("Hauptkomponente")
    axes[0].set_ylabel("Anteil erklärter Varianz")
    axes[0].yaxis.set_major_formatter(pct_formatter())
    axes[0].legend(fontsize=7)
    axes[1].plot(x, cum, "o-", color=PALETTE[0], ms=4)
    thr = cfg["reduction"]["pca_variance_threshold"]
    axes[1].axhline(thr, color="#666666", ls="--", lw=1)
    axes[1].axvline(n, color="#666666", ls=":", lw=1)
    axes[1].annotate(f"{n} Komponenten: {de_num(100 * cum[n - 1], 1)} %", (n, cum[n - 1]), xytext=(8, -22),
                     textcoords="offset points", fontsize=7.5)
    axes[1].set_xlabel("Anzahl Hauptkomponenten")
    axes[1].set_ylabel("Kumulierte erklärte Varianz")
    axes[1].yaxis.set_major_formatter(pct_formatter())
    for ax in axes:
        ax.set_xticks(x[::2])
    fig.tight_layout(w_pad=2.5)
    return save_fig(fig, cfg, "abb_07_pca_scree.png")


def pca_loadings(pca: dict, cfg: dict):
    """ladungen der ersten hauptkomponenten als heatmap."""
    n = cfg["reduction"]["pca_loadings_components"]
    L = pca["loadings"].iloc[:, :n]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    vmax = np.abs(L.to_numpy()).max()
    im = ax.imshow(L.to_numpy(), cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(L)), [_lab(cfg, c) for c in L.index], fontsize=7.5)
    ax.set_xticks(range(n), [f"HK {i + 1}\n({de_num(100 * pca['evr'][i], 1)} %)" for i in range(n)], fontsize=7.5)
    ax.grid(False)
    for i in range(L.shape[0]):
        for j in range(n):
            v = L.iat[i, j]
            ax.text(j, i, de_num(v, 2), ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, format=de_formatter())
    cb.set_label("Ladung")
    return save_fig(fig, cfg, "abb_08_pca_ladungen.png")


def mds_lle_quality(red: dict, cfg: dict):
    """mds-stress je zieldimension und lle-trustworthiness je nachbarschaftsgröße."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    dims = list(red["mds_stress"])
    axes[0].plot(dims, list(red["mds_stress"].values()), "o-", color=PALETTE[0], ms=5)
    axes[0].set_xlabel("Zieldimensionen (MDS)")
    axes[0].set_ylabel("Kruskal-Stress-1")
    axes[0].set_xticks(dims)
    axes[0].yaxis.set_major_formatter(de_formatter())
    nn = list(red["lle_error"])
    k_sel = cfg["reduction"]["trust_select_k"]
    tw = [red["trustworthiness"][f"LLE_nn{k}_k{k_sel}"] for k in nn]
    axes[1].plot(nn, tw, "o-", color=PALETTE[2], ms=5)
    axes[1].set_xlabel("Anzahl Nachbarn (LLE)")
    axes[1].set_ylabel(f"Trustworthiness (k = {k_sel})")
    axes[1].set_xticks(nn)
    axes[1].yaxis.set_major_formatter(de_formatter())
    return save_fig(fig, cfg, "abb_09_mds_lle_guete.png")


def embeddings_by_cluster(emb: dict, labels: np.ndarray, tw: dict, cfg: dict, pca_evr: np.ndarray):
    """2d-einbettungen von pca, mds und lle, eingefärbt nach cluster (pca-achsen mit erklärter varianz)."""
    k = len(np.unique(labels))
    k_tw = cfg["reduction"]["trust_select_k"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    for ax, (name, Y) in zip(axes, emb.items()):
        for c in range(k):
            m = labels == c
            ax.scatter(Y[m, 0], Y[m, 1], s=6, color=PALETTE[c], alpha=0.75, linewidths=0, marker=MARKERS[c],
                       label=f"Cluster {c + 1}")
        ax.set_title(f"{name} (T{k_tw} = {de_num(tw[f'{name}_k{k_tw}'], 3)})", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if name == "PCA":
            ax.set_xlabel(f"HK 1 ({de_num(100 * pca_evr[0], 1)} %)")
            ax.set_ylabel(f"HK 2 ({de_num(100 * pca_evr[1], 1)} %)")
        else:
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
    axes[0].legend(markerscale=3, fontsize=7, loc="lower center", bbox_to_anchor=(1.7, -0.3), ncol=k)
    return save_fig(fig, cfg, "abb_10_einbettungen_cluster.png")


# --- clustering

def baseline_elbow(scan: pd.DataFrame, scan_ws: pd.DataFrame, cfg: dict):
    """ellenbogen (wss, im jeweils eigenen raum) und silhouette im gemeinsamen raum: baseline gegen endmodell."""
    b = scan[scan.verfahren == "kmeans"]
    w = scan_ws[scan_ws.verfahren == "kmeans"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8))
    axes[0].plot(b.k, b.wss / b.wss.iloc[0], "o-", color=NEG_COLOR, ms=4, label="Iteration 1 (Baseline)")
    axes[0].plot(w.k, w.wss / w.wss.iloc[0], "s-", color=PALETTE[0], ms=4, label="Iteration 3 (final)")
    axes[0].set_ylabel("WSS (eigener Raum, relativ zu k = 2)")
    axes[0].yaxis.set_major_formatter(de_formatter())
    axes[1].plot(b.k, b.silhouette_ref, "o-", color=NEG_COLOR, ms=4, label="Iteration 1 (Baseline)")
    axes[1].plot(w.k, w.silhouette_ref, "s-", color=PALETTE[0], ms=4, label="Iteration 3 (final)")
    axes[1].set_ylabel("Silhouette (gemeinsamer Raum)")
    axes[1].yaxis.set_major_formatter(de_formatter())
    for ax in axes:
        ax.set_xlabel("Anzahl Cluster k")
        ax.set_xticks(b.k)
    axes[1].legend(fontsize=7)
    fig.tight_layout(w_pad=2.5)
    return save_fig(fig, cfg, "abb_11_baseline_ellenbogen.png")


def model_comparison(scan: pd.DataFrame, bic: pd.DataFrame, best_cov: str, cfg: dict):
    """silhouette und davies-bouldin der verfahren über k, rechts der bic-verlauf der gmm im erweiterten bereich."""
    sel = {"kmeans": ("k-Means", PALETTE[0], "o"), f"gmm_{best_cov}": (f"GMM ({best_cov})", PALETTE[1], "s"),
           "ward": ("Ward", PALETTE[2], "^")}
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    for key, (name, color, marker) in sel.items():
        s = scan[scan.verfahren == key]
        axes[0].plot(s.k, s.silhouette, marker=marker, color=color, ms=4, label=name)
        axes[1].plot(s.k, s.davies_bouldin, marker=marker, color=color, ms=4, label=name)
    for cov, color in zip(cfg["clustering"]["gmm_covariance_types"], PALETTE[3:]):
        s = bic[bic.kovarianz == cov].sort_values("k")
        axes[2].plot(s.k, s.bic / 1000, marker="o", ms=3, color=color, label=cov)
        i = s.bic.idxmin()
        axes[2].scatter(s.loc[i, "k"], s.loc[i, "bic"] / 1000, s=60, facecolors="none", edgecolors=color,
                        linewidths=1.5, zorder=3)
    axes[0].set_ylabel("Silhouette (höher = besser)")
    axes[1].set_ylabel("Davies-Bouldin (niedriger = besser)")
    axes[2].set_ylabel("BIC in Tsd. (niedriger = besser)")
    axes[0].legend(fontsize=6.5)
    # legende in den freien bereich zwischen der diag-kurve und den übrigen kurven
    axes[2].legend(fontsize=6.5, title="GMM (○ = Minimum)", title_fontsize=6.5, loc="upper right",
                   bbox_to_anchor=(1.0, 0.86))
    k_lo, k_hi = cfg["clustering"]["gmm_bic_k_range"]
    for ax in axes:
        ax.set_xlabel("k")
        ax.yaxis.set_major_formatter(de_formatter())
    for ax in axes[:2]:
        ax.set_xticks(sorted(scan.k.unique()))
    axes[2].set_xticks(range(k_lo, k_hi + 1, 4))
    fig.tight_layout()
    return save_fig(fig, cfg, "abb_12_modellvergleich.png")


def ward_dendrogram(Zlink: np.ndarray, k: int, cfg: dict):
    """dendrogramm der letzten fusionen mit schnittlinie bei k."""
    fig, ax = plt.subplots(figsize=(7.5, 3))
    cut = (Zlink[-k, 2] + Zlink[-k + 1, 2]) / 2
    dendrogram(Zlink, truncate_mode="lastp", p=30, ax=ax, color_threshold=cut, above_threshold_color="#888888",
               no_labels=True)
    ax.axhline(cut, color=NEG_COLOR, ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], cut, f" Schnitt: k = {k}", va="bottom", ha="right", fontsize=7.5, color=NEG_COLOR)
    ax.set_ylabel("Fusionsdistanz (Ward)")
    ax.set_xlabel("Zusammengefasste Teilnehmergruppen (letzte 30 Fusionen)")
    ax.yaxis.set_major_formatter(de_formatter())
    ax.grid(axis="x", visible=False)
    return save_fig(fig, cfg, "abb_13_dendrogramm.png")


def silhouette_plot(X: np.ndarray, labels: np.ndarray, cfg: dict):
    """silhouettenwerte je cluster des endmodells."""
    s = silhouette_samples(X, labels)
    k = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    y = 0
    for c in range(k):
        v = np.sort(s[labels == c])
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, color=PALETTE[c], alpha=0.9)
        ax.text(-0.05, y + len(v) / 2, f"C{c + 1}", ha="right", va="center", fontsize=8)
        y += len(v) + 15
    ax.axvline(s.mean(), color="#333333", ls="--", lw=1)
    ax.text(s.mean(), y, f" Mittel = {de_num(s.mean(), 3)}", fontsize=7.5, va="top")
    ax.set_xlabel("Silhouettenwert")
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(de_formatter())
    return save_fig(fig, cfg, "abb_14_silhouette_final.png")


def stability(boot: pd.DataFrame, ari: pd.DataFrame, cfg: dict):
    """bootstrap-ari je k und ari-matrix der verfahren beim finalen k."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3), gridspec_kw={"width_ratios": [1.3, 1]})
    ks = sorted(boot.k.unique())
    data = [boot[(boot.k == k) & (boot.verfahren == "kmeans")].ari for k in ks]
    bp = axes[0].boxplot(data, positions=ks, widths=0.5, patch_artist=True, showfliers=True,
                         flierprops=dict(marker="o", markersize=2.5, alpha=0.6))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE[4])
    final_k = cfg["clustering"]["final_k"]
    axes[0].axvline(final_k, color=NEG_COLOR, ls=":", lw=1)
    axes[0].set_xlabel("k (k-Means, Bootstrap-Resamples)")
    axes[0].set_ylabel("Adjusted Rand Index")
    axes[0].yaxis.set_major_formatter(de_formatter())
    axes[1].imshow(ari.to_numpy(dtype=float), cmap=SEQ_CMAP, vmin=0, vmax=1)
    axes[1].set_xticks(range(len(ari)), ari.columns, rotation=45, ha="right", fontsize=7)
    axes[1].set_yticks(range(len(ari)), ari.index, fontsize=7)
    axes[1].grid(False)
    for i in range(len(ari)):
        for j in range(len(ari)):
            v = float(ari.iat[i, j])
            axes[1].text(j, i, de_num(v, 2), ha="center", va="center", fontsize=6.5,
                         color="black" if v > 0.6 else "white")
    axes[1].set_title(f"Übereinstimmung der Verfahren (k = {final_k}, ARI)", fontsize=8)
    fig.tight_layout()
    return save_fig(fig, cfg, "abb_15_stabilitaet.png")


# --- clusterprofile

def cluster_heatmap(means: pd.DataFrame, sizes: pd.DataFrame, cfg: dict):
    """clustermittel je merkmal (0-1-skala), farbe = abweichung vom gesamtmittel."""
    m = means.drop(index="gesamt")
    dev = m - means.loc["gesamt"]
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    vmax = np.abs(dev.to_numpy()).max()
    im = ax.imshow(dev.T.to_numpy(), cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    cols = [f"Cluster {c}\n(n = {de_num(n, 0)})" for c, n in zip(sizes.cluster, sizes.n)] + ["Gesamt"]
    ax.set_xticks(range(len(cols)), cols, fontsize=7.5)
    ax.set_yticks(range(m.shape[1]), [_lab(cfg, c) for c in m.columns], fontsize=7.5)
    ax.grid(False)
    for i, c in enumerate(m.columns):
        for j, cl in enumerate(m.index):
            ax.text(j, i, de_num(m.loc[cl, c], 2), ha="center", va="center", fontsize=7,
                    color="white" if abs(dev.loc[cl, c]) > 0.6 * vmax else "black")
        ax.text(len(m.index), i, de_num(means.loc["gesamt", c], 2), ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, format=de_formatter())
    cb.set_label("Abweichung vom Gesamtmittel (0–1-Skala)")
    return save_fig(fig, cfg, "abb_16_cluster_heatmap.png")


def cluster_profiles(means: pd.DataFrame, top: dict, cfg: dict):
    """je cluster die trennschärfsten merkmale als abweichung vom gesamtmittel."""
    clusters = [c for c in means.index if c != "gesamt"]
    ncols = 2
    nrows = int(np.ceil(len(clusters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, 2.4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    lim = 0
    for ax, c in zip(axes, clusters):
        feats = top[c][::-1]
        dev = (means.loc[c, feats] - means.loc["gesamt", feats])
        lim = max(lim, dev.abs().max())
        ax.barh([_lab(cfg, f) for f in feats], dev, color=[PALETTE[0] if v > 0 else NEG_COLOR for v in dev],
                height=0.65)
        ax.axvline(0, color="#333333", lw=0.8)
        ax.set_title(f"Cluster {c}", fontsize=9, color=PALETTE[c - 1], loc="left", fontweight="bold")
        ax.tick_params(axis="y", labelsize=6.8)
        ax.grid(axis="y", visible=False)
    for ax in axes[len(clusters):]:
        ax.axis("off")
    for ax in axes:
        ax.set_xlim(-lim * 1.1, lim * 1.1)
        ax.xaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(0.5))
        ax.xaxis.set_major_formatter(de_formatter())
    for ax in axes[-ncols:]:
        ax.set_xlabel("Abweichung vom Gesamtmittel")
    fig.tight_layout()
    return save_fig(fig, cfg, "abb_17_cluster_profile.png")


def profile_variables(shares: pd.DataFrame, counts: pd.DataFrame, cfg: dict):
    """anteile ausgewählter profilvariablen je cluster, zellen mit n < min_cell_n unterdrückt."""
    min_n = cfg["profiling"]["min_cell_n"]
    k = shares.shape[1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(shares))
    h = 0.8 / k
    for j, c in enumerate(shares.columns):
        vals = shares[c].where(counts[c] >= min_n)
        ax.barh(y + (j - (k - 1) / 2) * h, vals.fillna(0), height=h * 0.92, color=PALETTE[j], label=f"Cluster {c}")
        for i, (v, n) in enumerate(zip(shares[c], counts[c])):
            if n < min_n:
                ax.text(0.005, y[i] + (j - (k - 1) / 2) * h, f"< {min_n}", va="center", fontsize=5.5)
    ax.set_yticks(y, shares.index, fontsize=7.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(pct_formatter())
    ax.set_xlabel("Anteil innerhalb des Clusters")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, ncol=k, loc="lower center", bbox_to_anchor=(0.4, 1.0))
    return save_fig(fig, cfg, "abb_18_profilvariablen.png")


def tfidf_terms(terms: pd.DataFrame, cfg: dict):
    """clusterspezifische terme aus den freitexten."""
    clusters = sorted(terms.cluster.unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(7.5, 2.8))
    for ax, c in zip(np.atleast_1d(axes), clusters):
        t = terms[terms.cluster == c].sort_values("rang", ascending=False)
        ax.barh(t.term, t.tfidf_diff * 100, color=PALETTE[c - 1], height=0.65)
        ax.set_title(f"Cluster {c}", fontsize=8.5, color=PALETTE[c - 1], loc="left", fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_formatter(de_formatter())
        ax.grid(axis="y", visible=False)
    fig.supxlabel("TF-IDF-Differenz zum Rest (× 100)", fontsize=8)
    fig.tight_layout()
    return save_fig(fig, cfg, "abb_19_tfidf_terme.png")


def index_space(idx: pd.DataFrame, labels: np.ndarray, cfg: dict):
    """lage der cluster im raum aus unterstützungs- und offenheitsindex."""
    rng = np.random.default_rng(cfg["random_state"])
    k = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    # leichtes rauschen, weil viele personen exakt dieselben indexwerte haben
    x = idx["support_index"] + rng.uniform(-0.02, 0.02, len(idx))
    y = idx["openness_index"] + rng.uniform(-0.03, 0.03, len(idx))
    for c in range(k):
        m = labels == c
        ax.scatter(x[m], y[m], s=5, alpha=0.35, color=PALETTE[c], linewidths=0, marker=MARKERS[c])
    for c in range(k):
        m = labels == c
        cx, cy = idx.loc[m, "support_index"].mean(), idx.loc[m, "openness_index"].mean()
        ax.scatter(cx, cy, s=140, color=PALETTE[c], edgecolor="white", linewidth=2, zorder=3, marker=MARKERS[c])
        ax.annotate(f"C{c + 1}", (cx, cy), xytext=(8, 6), textcoords="offset points", fontsize=9, fontweight="bold")
    ax.axvline(idx["support_index"].mean(), color="#888888", ls=":", lw=1)
    ax.axhline(idx["openness_index"].mean(), color="#888888", ls=":", lw=1)
    ax.set_xlabel("Unterstützungsindex (Angebote, 0–1)")
    ax.set_ylabel("Offenheitsindex (0–1)")
    ax.xaxis.set_major_formatter(de_formatter())
    ax.yaxis.set_major_formatter(de_formatter())
    return save_fig(fig, cfg, "abb_20_indexraum.png")


def baseline_space(Y: np.ndarray, labels: np.ndarray, tw: float, cfg: dict):
    """pca-projektion aller übrigen umfragemerkmale (baseline-matrix), eingefärbt nach finalem cluster."""
    k = len(np.unique(labels))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for c in range(k):
        m = labels == c
        ax.scatter(Y[m, 0], Y[m, 1], s=6, color=PALETTE[c], alpha=0.7, linewidths=0, marker=MARKERS[c],
                   label=f"Cluster {c + 1}")
    ax.set_xlabel("Hauptkomponente 1 (alle Merkmale)")
    ax.set_ylabel("Hauptkomponente 2 (alle Merkmale)")
    ax.set_title(f"T{cfg['reduction']['trust_select_k']} = {de_num(tw, 3)}", fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(markerscale=2.5, fontsize=7, loc="best")
    return save_fig(fig, cfg, "abb_21_gesamtdaten_pca.png")
