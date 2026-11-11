"""Download, load and clean the public datasets.

The UCI concrete file has long column headers, repeated mixes tested at several ages, and a few
exact duplicate rows. `load_concrete` turns it into a tidy frame with short names and a
`mix_id` so the emulator can be validated with grouped cross-validation (no leakage between
ages of the same mix).
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
OPENML_DATA_ID = 44959
EXPECTED_ROWS = 1030
UCI_XLS_NAME = "Concrete_Data.xls"
OPENML_CSV_NAME = "concrete_openml.csv"

INGREDIENTS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
]
AGE = "age_days"
TARGET = "strength_mpa"
SHORT_NAMES = [*INGREDIENTS, AGE, TARGET]

# Fixed mapping from normalized long headers (lowercase, single spaces) to short names.
# Covers the UCI Excel headers and the OpenML 44959 column names.
CONCRETE_COLUMNS: dict[str, str] = {
    "cement (component 1)(kg in a m^3 mixture)": "cement",
    "blast furnace slag (component 2)(kg in a m^3 mixture)": "slag",
    "fly ash (component 3)(kg in a m^3 mixture)": "fly_ash",
    "water (component 4)(kg in a m^3 mixture)": "water",
    "superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",
    "coarse aggregate (component 6)(kg in a m^3 mixture)": "coarse_agg",
    "fine aggregate (component 7)(kg in a m^3 mixture)": "fine_agg",
    "age (day)": "age_days",
    "concrete compressive strength(mpa, megapascals)": "strength_mpa",
    # OpenML names
    "cement": "cement",
    "blast_furnace_slag": "slag",
    "fly_ash": "fly_ash",
    "water": "water",
    "superplasticizer": "superplasticizer",
    "coarse_aggregate": "coarse_agg",
    "fine_aggregate": "fine_agg",
    "age": "age_days",
    "strength": "strength_mpa",
    "concrete_compressive_strength": "strength_mpa",
}


def _normalize_header(name: str) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().lower()


def rename_concrete_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename UCI or OpenML headers to the 9 short names, in a fixed order."""
    mapping = {}
    for col in df.columns:
        key = _normalize_header(col)
        if key not in CONCRETE_COLUMNS:
            raise KeyError(f"Unexpected concrete column header: {col!r}")
        mapping[col] = CONCRETE_COLUMNS[key]
    out = df.rename(columns=mapping)
    if sorted(out.columns) != sorted(SHORT_NAMES):
        raise ValueError(f"Expected 9 concrete columns {SHORT_NAMES}, got {list(out.columns)}")
    return out[SHORT_NAMES].astype(float)


def add_mix_id(
    df: pd.DataFrame, ingredient_cols: list[str] | None = None, round_to: float = 0.1
) -> pd.DataFrame:
    """Add `mix_id`: a short hash of the ingredient amounts rounded to `round_to` kg/m3.

    The same recipe tested at 3, 7, 28 and 90 days gets the same id, which is what grouped
    cross-validation needs.
    """
    cols = ingredient_cols or INGREDIENTS
    rounded = np.round(df[cols].to_numpy(dtype=float) / round_to) * round_to

    def _hash(row: np.ndarray) -> str:
        text = ",".join(f"{v:.1f}" for v in row)
        return hashlib.sha1(text.encode("ascii")).hexdigest()[:12]

    out = df.copy()
    out["mix_id"] = [_hash(r) for r in rounded]
    return out


def clean_concrete(df_raw: pd.DataFrame, round_to: float = 0.1) -> tuple[pd.DataFrame, int]:
    """Rename, drop exact duplicate rows and add `mix_id`. Returns (frame, n_dropped)."""
    df = rename_concrete_columns(df_raw)
    if df.isna().any().any():
        raise ValueError("Concrete data has missing values; zeros are real, NaN is not expected")
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    return add_mix_id(df, round_to=round_to), n_dropped


def _read_xls(path: Path) -> pd.DataFrame:
    """Read the Excel 97 file with xlrd (the format the UCI zip ships).

    pandas 3 warns that its xlrd engine is deprecated in favour of calamine. The warning is
    caught here, for this one call only, because xlrd is the documented dependency for `.xls`.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*xlrd engine is deprecated.*")
        return pd.read_excel(path, engine="xlrd")


def read_concrete_raw(raw_dir: Path) -> pd.DataFrame:
    """Read the raw concrete table from the UCI Excel file, or the OpenML CSV fallback."""
    raw_dir = Path(raw_dir)
    xls = raw_dir / UCI_XLS_NAME
    if xls.exists():
        return _read_xls(xls)
    csv = raw_dir / OPENML_CSV_NAME
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"No concrete data in {raw_dir}. Run: uv run python -m formopt download"
    )


def load_concrete(
    raw_dir: Path, processed_dir: Path | None = None, round_to: float = 0.1
) -> pd.DataFrame:
    """Load the concrete data: short names, duplicates dropped (count logged), `mix_id` added.

    If `processed_dir` is given, a copy is saved to `processed_dir/concrete.csv`.
    """
    df, n_dropped = clean_concrete(read_concrete_raw(raw_dir), round_to=round_to)
    logger.info(
        "Concrete: dropped %d exact duplicate rows, %d rows left, %d unique mixes",
        n_dropped,
        len(df),
        df["mix_id"].nunique(),
    )
    if processed_dir is not None:
        processed_dir = Path(processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_dir / "concrete.csv", index=False)
    return df


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch(url: str, dest: Path, timeout: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "formopt-download/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 (fixed URL)
        dest.write_bytes(resp.read())
    return dest


def download_concrete(raw_dir: Path, timeout: float = 30.0, force: bool = False) -> Path:
    """Download the UCI concrete zip and unzip it; fall back to OpenML if UCI fails.

    Idempotent: if the data file is already there (and `force` is False) nothing is fetched.
    Returns the path of the data file that `load_concrete` will read.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    xls = raw_dir / UCI_XLS_NAME
    if xls.exists() and not force:
        logger.info("Already present: %s (sha256 %s)", xls, _sha256(xls)[:16])
        return xls
    zip_path = raw_dir / "concrete.zip"
    uci_error: Exception | None = None
    try:
        if not zip_path.exists() or force:
            _fetch(UCI_ZIP_URL, zip_path, timeout)
        logger.info("UCI zip sha256 %s", _sha256(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        if not xls.exists():
            raise FileNotFoundError(f"{UCI_XLS_NAME} not found inside {zip_path}")
        _check_rows(_read_xls(xls))
        return xls
    except Exception as exc:  # noqa: BLE001 (any failure triggers the fallback)
        uci_error = exc
        logger.warning("UCI download failed (%s); trying OpenML", exc)
    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(
            data_id=OPENML_DATA_ID, as_frame=True, data_home=str(raw_dir.parent / "openml")
        )
        frame = bunch.frame
        _check_rows(frame)
        out = raw_dir / OPENML_CSV_NAME
        frame.to_csv(out, index=False)
        return out
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not download the concrete data from UCI or OpenML.\n"
            f"UCI error: {uci_error}\nOpenML error: {exc}\n"
            "Manual fix: download the zip in a browser from "
            "https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength and save it "
            f"to {zip_path}, then run this command again."
        ) from exc


def _check_rows(frame: pd.DataFrame) -> None:
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} concrete rows, got {len(frame)}")
