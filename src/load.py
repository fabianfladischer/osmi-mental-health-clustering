"""rohdaten laden, prüfen und die spalten in kurzcodes umbenennen."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd

from .utils import ROOT


def download(cfg: dict) -> Path:
    """lädt den datensatz über kagglehub, falls die csv noch nicht lokal liegt."""
    target = ROOT / cfg["data"]["raw_file"]
    if target.exists():
        return target
    import kagglehub  # nur nötig, wenn die csv fehlt

    src_dir = Path(kagglehub.dataset_download(cfg["data"]["kaggle_handle"]))
    src = src_dir / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, target)
    return target


def sha256(path: Path) -> str:
    """sha256-prüfsumme einer datei."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """liest die csv, prüft die form und benennt die spalten laut codebook um."""
    path = download(cfg)
    df = pd.read_csv(path)
    expected = tuple(cfg["data"]["expected_shape"])
    if df.shape != expected:
        raise ValueError(f"Unerwartete Form {df.shape}, erwartet {expected}")

    codebook = cfg["codebook"]
    if len(codebook) != df.shape[1]:
        raise ValueError("Codebook passt nicht zur Spaltenzahl")
    questions = {}
    for entry, col in zip(codebook, df.columns):
        if not col.strip().startswith(entry["prefix"]):
            raise ValueError(f"Spalte '{col[:60]}' passt nicht zu Code {entry['code']}")
        # pandas hängt bei doppelten spaltennamen ".1" an, für das codebook wieder entfernen
        questions[entry["code"]] = re.sub(r"\.\d+$", "", col.strip())
    df.columns = [e["code"] for e in codebook]
    return df, questions
