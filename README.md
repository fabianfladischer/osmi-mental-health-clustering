# Clusteranalyse OSMI Mental Health in Tech Survey 2016

Dieses Repository enthält den Code zu meiner Fallstudie im Kurs **DLBDSMLUSL01_D – Maschinelles Lernen – Unsupervised Learning und Feature Engineering** an der IU Internationale Hochschule (B.Sc. FinTech).

In der Fallstudie werden Beschäftigte aus Tech-Berufen anhand ihrer Wahrnehmung des Arbeitsplatzes (Angebote des Arbeitgebers, Arbeitsklima, Stigma) mit Verfahren des Unsupervised Learning in Gruppen eingeteilt. Aus den Gruppen leite ich anschließend Ansatzpunkte für ein betriebliches Präventionsprogramm ab.

## Vorgehen

1. **Datenaufbereitung:**
   - Selbstständige und Personen ohne Tech-Bezug werden ausgeschlossen (n = 1.131).
   - Alter und Geschlecht werden bereinigt.
   - Die 15 Arbeitsplatz-Items werden auf eine Skala von 0 bis 1 kodiert, dazu kommt ein Unsicherheitsindex für „Weiß nicht“-Antworten.
2. **Dimensionsreduktion:**
   - PCA auf der Kovarianzmatrix (9 Hauptkomponenten, 83,5 % Varianz).
   - Zum Vergleich MDS und LLE, bewertet über die Trustworthiness.
3. **Clusteranalyse:**
   - k-Means, GMM und Ward-Verfahren, jeweils für mehrere k.
   - Das Endmodell ist k-Means mit k = 4.
   - Die Stabilität wird über Bootstrap und den Adjusted Rand Index geprüft.
4. **Profilierung:**
   - Beschreibung der Cluster über die Clustermittelwerte.
   - Chi²- und Kruskal-Wallis-Tests für die Profilierungsvariablen.
   - TF-IDF auf den Freitextantworten.

Demografische und gesundheitliche Merkmale fließen nicht in das Clustering ein, sondern werden nur zur Beschreibung der Cluster verwendet. Zellen mit weniger als 10 Personen werden nicht ausgewiesen.

## Daten

Verwendet wird der Datensatz *OSMI Mental Health in Tech Survey 2016* von Open Sourcing Mental Illness. Er ist auf Kaggle unter der Lizenz CC BY-SA 4.0 verfügbar:

https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016

Die CSV-Datei muss unter `data/raw/` liegen. Falls sie fehlt, kann sie über den Link oben heruntergeladen werden.

## Installation

Getestet mit Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate        # unter Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ausführen

```bash
python run_all.py
```

Die Laufzeit liegt bei etwa 6 Minuten (die Hauptpipeline allein braucht ca. 4 Minuten, der Rest entfällt auf die ergänzenden Analysen aus dem Anhang).

- Alle Abbildungen landen in `figures/`, die Tabellen und Kennzahlen in `results/`.
- Die Parameter stehen in `config/pipeline.yaml`, z. B. die Schwellenwerte für die Feature-Auswahl oder die Anzahl der Cluster.
- Der Zufallsstartwert ist überall `random_state = 42`, dadurch kommen bei jedem Durchlauf dieselben Ergebnisse heraus.

Die Tests laufen mit:

```bash
  pip install -r requirements-dev.txt
  pytest
```

## Aufbau

```text
config/pipeline.yaml   Parameter der Pipeline
src/                   Module für Laden, Bereinigung, Features, Dimensionsreduktion, Clustering, Profile und Grafiken
notebooks/             Notebook, das die wichtigsten Schritte nacheinander zeigt
tests/                 Tests (pytest)
figures/               Abbildungen aus der Fallstudie
results/               Tabellen und Kennzahlen (CSV, JSON)
run_all.py             startet die komplette Pipeline
data/raw/              Rohdaten (CSV, nicht versioniert)
```

Das vollständige Codebook (in der Fallstudie gekürzt als Tabelle 6) liegt unter `results/tab_codebook.csv`.

## Hinweise zu den Ergebnissen

- Die Umfrage wurde 2016 freiwillig online durchgeführt und ist stark US-geprägt. Die Ergebnisse sind deshalb nicht repräsentativ.
- Die Silhouette des Endmodells ist mit 0,158 niedrig. Die vier Cluster sind daher eher eine praktische Einteilung für die Maßnahmenplanung als klar getrennte Gruppen.
- Alle Zusammenhänge sind rein beschreibend (Querschnittsdaten), kausale Aussagen sind nicht möglich.

Fabian Fladischer, 2026
