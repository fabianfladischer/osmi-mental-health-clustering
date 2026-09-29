"""hilfsfunktionen für konfiguration, kennzahlen, pfade und grafikstil."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

# okabe-ito-farben, reihenfolge so, dass benachbarte farben auch bei farbsehschwäche unterscheidbar bleiben
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
NEG_COLOR = "#D55E00"
SEQ_CMAP = "cividis"
DIV_CMAP = "PuOr_r"


def load_config(path: str | Path | None = None) -> dict:
    """lädt config/pipeline.yaml."""
    path = Path(path) if path else ROOT / "config" / "pipeline.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def out_path(cfg: dict, kind: str, name: str) -> Path:
    """pfad in figures/ oder results/, der ordner wird bei bedarf angelegt."""
    p = ROOT / cfg["paths"][kind] / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_metrics(metrics: dict, path: Path) -> None:
    """kennzahlen als json speichern (sortiert, umlaute bleiben lesbar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(metrics), f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)


def _to_builtin(v):
    """numpy-typen in normale python-typen umwandeln, sonst klappt json.dump nicht."""
    if isinstance(v, dict):
        return {str(k): _to_builtin(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_builtin(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        # nan wird zu null (gültiges json), sonst reichen 6 signifikante stellen
        return None if math.isnan(v) else float(f"{float(v):.6g}")
    if isinstance(v, np.ndarray):
        return _to_builtin(v.tolist())
    return v


@contextmanager
def timer(metrics: dict, key: str):
    """misst die laufzeit eines blocks in sekunden."""
    t0 = time.perf_counter()
    yield
    metrics.setdefault("laufzeit", {})[key] = time.perf_counter() - t0


def de_num(x: float, digits: int = 2) -> str:
    """zahl mit dezimalkomma und tausenderpunkt formatieren."""
    s = f"{x:,.{digits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def set_style() -> None:
    """einheitlicher stil für alle abbildungen."""
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.formatter.use_locale": False,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "legend.frameon": False,
    })


class _DeScalarFormatter(mpl.ticker.ScalarFormatter):
    """normale tick-beschriftung von matplotlib, nur mit dezimalkomma."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def de_formatter():
    """achsenbeschriftung mit dezimalkomma, stellen richten sich nach dem tickabstand."""
    f = _DeScalarFormatter(useOffset=False)
    f.set_scientific(False)
    return f


def pct_formatter(digits: int = 0):
    """achsenbeschriftung in prozent für werte zwischen 0 und 1."""
    return mpl.ticker.FuncFormatter(lambda x, _: de_num(100 * x, digits) + " %")


def save_fig(fig, cfg: dict, name: str) -> Path:
    """abbildung als png speichern und schließen."""
    p = out_path(cfg, "figures", name)
    fig.savefig(p)
    plt.close(fig)
    return p
