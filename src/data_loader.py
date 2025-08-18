# src/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, IO
from zipfile import BadZipFile

DATE_COL_CANDIDATES = ["Date","date","Txn Date","Transaction Date"]
YEAR_COLS = ["Year","YEAR","year"]
MONTH_COLS = ["Month","MONTH","month","Mon"]
DAY_COLS = ["Day","DAY","day","Dom"]
STORE_COLS = ["Store No","Store","store","Branch","Location"]
CHANNEL_COLS = ["Channel","channel","Country","Country Code"]
GROUP_COLS = ["Group Name","Group","Budget Category Name","Subgroup","Sub Group","Category","Sub-Category"]
ITEM_COLS = ["Item Description","Item","SKU","Barcode","EAN","UPC","Product","Material"]
RECEIPT_COLS = ["Receipt No","Receipt","Invoice No","Bill No","Transaction No","Order No"]
SALES_QTY_COLS = ["Sales Qty","Qty","Quantity","Units"]
SALES_VAL_COLS = ["Sales Val","Sales Value","Net Sales","Sales"]

def _find_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    lc = [c.lower() for c in cols]
    for cand in candidates:
        cand = cand.lower()
        for i, c in enumerate(lc):
            if cand == c or cand in c:
                return cols[i]
    return None

def _check_path_health(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"File is empty: {path}")
    # Detect Git LFS pointer (tiny text file)
    try:
        head = path.read_bytes()[:256]
        if b"git-lfs" in head and b"oid sha256" in head:
            raise ValueError(
                f"'{path.name}' looks like a Git LFS pointer. "
                "Upload the real file via the uploader or ensure LFS files are fetched on the host."
            )
    except Exception:
        pass

def _read_any_tabular(obj: Union[str, Path, IO[bytes]]):
    """Return dict of DataFrames (sheet_name -> df). Supports CSV and Excel.
       Gives actionable errors for empty/LFS/misnamed files."""
    if isinstance(obj, (str, Path)):
        p = Path(obj)
        _check_path_health(p)
        name = p.name.lower()
        handle = str(p)
    else:
        name = getattr(obj, "name", "uploaded").lower()
        handle = obj

    # CSV explicit
    if name.endswith(".csv"):
        df = pd.read_csv(handle)
        if df.shape[1] == 0:
            raise ValueError("CSV has no columns. Is the file empty or corrupted?")
        return {"CSV": df}

    # Excel explicit
    if name.endswith(".xlsx"):
        try:
            return pd.read_excel(handle, sheet_name=None, engine="openpyxl")
        except BadZipFile:
            # Misnamed CSV/empty file with .xlsx extension
            df = pd.read_csv(handle)
            if df.shape[1] == 0:
                raise ValueError("File named .xlsx is not a real Excel (and CSV fallback has no columns).")
            return {"CSV": df}
    if name.endswith(".xls"):
        return pd.read_excel(handle, sheet_name=None, engine="xlrd")

    # Unknown: try Excel then CSV
    try:
        return pd.read_excel(handle, sheet_name=None, engine="openpyxl")
    except BadZipFile:
        df = pd.read_csv(handle)
        if df.shape[1] == 0:
            raise ValueError("File is neither valid Excel nor CSV with columns.")
        return {"CSV": df}

def load_excel(path_or_buffer: Union[str, Path, IO[bytes]]) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    sheets = _read_any_tabular(path_or_buffer)
    sheet_name = max(sheets, key=lambda k: sheets[k].shape[0])
    df = sheets[sheet_name].copy()
    cols = list(df.columns)

    date_col = _find_col(cols, DATE_COL_CANDIDATES)
    ycol = _find_col(cols, YEAR_COLS)
    mcol = _find_col(cols, MONTH_COLS)
    dcol = _find_col(cols, DAY_COLS)

    if date_col is None and ycol and mcol:
        d = df[[ycol, mcol]].copy()
        d["__day"] = df[dcol] if dcol and dcol in df else 1
        df["__date"] = pd.to_datetime(
            d.rename(columns={ycol: "__year", mcol: "__month"})[["__year", "__month", "__day"]],
            errors="coerce"
        )
        date_col = "__date"
    elif date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    meta = {
        "date": date_col,
        "store": _find_col(cols, STORE_COLS),
        "channel": _find_col(cols, CHANNEL_COLS),
        "group": _find_col(cols, GROUP_COLS),
        "item": _find_col(cols, ITEM_COLS),
        "receipt": _find_col(cols, RECEIPT_COLS),
        "qty": _find_col(cols, SALES_QTY_COLS),
        "value": _find_col(cols, SALES_VAL_COLS),
    }

    if meta["qty"] and meta["value"]:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["__unit_price"] = df[meta["value"]].astype(float) / df[meta["qty"]].astype(float).replace(0, np.nan)

    return df, meta

def monthly_aggregate(df: pd.DataFrame, date_col: str, value_col: str, group_cols=None) -> pd.DataFrame:
    gcols = group_cols or []
    d = df[df[date_col].notna()].copy()
    d["month"] = d[date_col].values.astype("datetime64[M]")
    agg = d.groupby(gcols + ["month"], as_index=False)[value_col].sum()
    return agg
