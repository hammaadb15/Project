from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

import pandas as pd
import requests

# PDF export (ReportLab)
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


# -----------------------------
# 1) Load data (CSV / XLS / XLSX)
# -----------------------------
def load_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Requirement: "Be able to import data from CSV or XLS data formats" :contentReference[oaicite:1]{index=1}
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(file_path)

    raise ValueError("Unsupported file type. Please provide .csv, .xls, or .xlsx")


# -----------------------------
# 2) Date parsing (helper)
# -----------------------------
def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Parses a date column safely (dayfirst=True helps UK-style dates).
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True).dt.tz_localize(None)
    return out


# -----------------------------
# 3) Keyword search across ALL columns
# -----------------------------
def keyword_search_filter(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """
    Requirement: "Allow keyword searching, to look across all data and filter by keyword" :contentReference[oaicite:2]{index=2}
    """
    if not keyword or not str(keyword).strip():
        return df

    kw = str(keyword).strip().lower()
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        mask |= df[col].astype(str).str.lower().str.contains(kw, na=False)
    return df[mask].copy()


# -----------------------------
# 4) Dynamic filters (date, category/subcategory, location)
# -----------------------------
def filter_by_date(
    df: pd.DataFrame,
    date_col: str,
    mode: str = "none",  # "monthly" | "quarterly" | "yearly" | "range" | "none"
    year: Optional[int] = None,
    month: Optional[int] = None,      # 1-12
    quarter: Optional[int] = None,    # 1-4
    start_date: Optional[Union[str, date, datetime]] = None,
    end_date: Optional[Union[str, date, datetime]] = None,
) -> pd.DataFrame:
    """
    Requirement: filter by Date (Monthly, Quarterly, Yearly, manual date range) :contentReference[oaicite:3]{index=3}
    """
    if mode is None:
        mode = "none"
    mode = mode.strip().lower()

    out = df.copy()
    out = out[out[date_col].notna()].copy()

    if mode == "none":
        return df.copy()

    if mode == "yearly":
        if year is None:
            raise ValueError("year is required for yearly mode")
        return out[out[date_col].dt.year == int(year)].copy()

    if mode == "monthly":
        if year is None or month is None:
            raise ValueError("year and month are required for monthly mode")
        return out[(out[date_col].dt.year == int(year)) & (out[date_col].dt.month == int(month))].copy()

    if mode == "quarterly":
        if year is None or quarter is None:
            raise ValueError("year and quarter are required for quarterly mode")
        q = int(quarter)
        if q not in (1, 2, 3, 4):
            raise ValueError("quarter must be 1..4")
        months = {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9), 4: (10, 11, 12)}[q]
        return out[(out[date_col].dt.year == int(year)) & (out[date_col].dt.month.isin(months))].copy()

    if mode == "range":
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for range mode")
        s = pd.to_datetime(start_date)
        e = pd.to_datetime(end_date)
        return out[(out[date_col] >= s) & (out[date_col] <= e)].copy()

    raise ValueError("mode must be one of: none, yearly, monthly, quarterly, range")


def filter_by_category(
    df: pd.DataFrame,
    category_col: Optional[str] = None,
    subcategory_col: Optional[str] = None,
    categories: Optional[List[str]] = None,
    subcategories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Requirement: filter by Seizure Category and Sub-category :contentReference[oaicite:4]{index=4}
    """
    out = df.copy()

    if category_col and categories:
        out = out[out[category_col].astype(str).isin([str(x) for x in categories])].copy()

    if subcategory_col and subcategories:
        out = out[out[subcategory_col].astype(str).isin([str(x) for x in subcategories])].copy()

    return out


def filter_by_location(
    df: pd.DataFrame,
    postcode_col: Optional[str] = None,
    city_col: Optional[str] = None,
    postcodes: Optional[List[str]] = None,
    cities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Requirement: filter by Location (Post code, City) :contentReference[oaicite:5]{index=5}
    """
    out = df.copy()

    if postcode_col and postcodes:
        wanted = [normalize_uk_postcode(p) for p in postcodes]
        out = out[out[postcode_col].astype(str).apply(normalize_uk_postcode).isin(wanted)].copy()

    if city_col and cities:
        wanted = [str(c).strip().lower() for c in cities]
        out = out[out[city_col].astype(str).str.strip().str.lower().isin(wanted)].copy()

    return out


# -----------------------------
# 5) Top 10 summary
# -----------------------------
def top_n_summary(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    """
    Requirement: "Produce a top 10 summary of data" :contentReference[oaicite:6]{index=6}
    Example: top 10 most seized items.
    """
    vc = (
        df[column]
        .astype(str)
        .fillna("")
        .replace("nan", "")
        .value_counts()
        .head(n)
        .reset_index()
    )
    vc.columns = [column, "count"]
    return vc


# -----------------------------
# 6) Geocoding (postcode/city -> lat/lon) with caching
# -----------------------------
CACHE_PATH = Path(".geocode_cache.json")
_POSTCODE_RE = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}$", re.IGNORECASE)


def _load_cache() -> Dict[str, Dict[str, float]]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Dict[str, float]]) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def normalize_uk_postcode(pc: str) -> str:
    if pc is None:
        return ""
    pc = str(pc).strip().upper()
    pc = re.sub(r"\s+", "", pc)
    if len(pc) <= 3:
        return pc
    return pc[:-3] + " " + pc[-3:]


def looks_like_uk_postcode(pc: str) -> bool:
    pc = normalize_uk_postcode(pc)
    return bool(_POSTCODE_RE.match(pc))


def geocode_uk_postcode(postcode: str, timeout: int = 10) -> Optional[Tuple[float, float]]:
    """
    Uses postcodes.io for UK postcode -> (lat, lon). No API key.
    """
    if not postcode:
        return None
    postcode = normalize_uk_postcode(postcode)
    key = f"pc:{postcode}"

    cache = _load_cache()
    if key in cache:
        return cache[key]["lat"], cache[key]["lon"]

    url = f"https://api.postcodes.io/postcodes/{requests.utils.quote(postcode)}"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != 200 or not data.get("result"):
            return None
        lat = data["result"]["latitude"]
        lon = data["result"]["longitude"]
        if lat is None or lon is None:
            return None
        cache[key] = {"lat": float(lat), "lon": float(lon)}
        _save_cache(cache)
        return float(lat), float(lon)
    except Exception:
        return None


def geocode_city(city: str, country: str = "United Kingdom", timeout: int = 12) -> Optional[Tuple[float, float]]:
    """
    Uses OpenStreetMap Nominatim for city -> (lat, lon).
    """
    if not city:
        return None
    city = str(city).strip()
    key = f"city:{city.lower()}|{country.lower()}"

    cache = _load_cache()
    if key in cache:
        return cache[key]["lat"], cache[key]["lon"]

    # small delay to be polite
    time.sleep(1.0)

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{city}, {country}", "format": "json", "limit": 1}
    headers = {"User-Agent": "Seizure-Data-Visualisation-Tool/1.0 (educational)"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        res = r.json()
        if not res:
            return None
        lat = float(res[0]["lat"])
        lon = float(res[0]["lon"])
        cache[key] = {"lat": lat, "lon": lon}
        _save_cache(cache)
        return lat, lon
    except Exception:
        return None


def attach_lat_lon(
    df: pd.DataFrame,
    postcode_col: Optional[str] = None,
    city_col: Optional[str] = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """
    Requirement: heat map based on location data such as postcode :contentReference[oaicite:7]{index=7}
    This prepares the backend data for the heatmap by ensuring lat/lon exist.
    """
    out = df.copy()
    if lat_col not in out.columns:
        out[lat_col] = pd.NA
    if lon_col not in out.columns:
        out[lon_col] = pd.NA

    for i in out.index:
        if pd.notna(out.at[i, lat_col]) and pd.notna(out.at[i, lon_col]):
            continue

        pc = None
        city = None
        if postcode_col and postcode_col in out.columns:
            pc = out.at[i, postcode_col]
            pc = "" if pd.isna(pc) else str(pc)
            pc = normalize_uk_postcode(pc)

        if city_col and city_col in out.columns:
            city = out.at[i, city_col]
            city = "" if pd.isna(city) else str(city).strip()

        coords = None
        if pc and looks_like_uk_postcode(pc):
            coords = geocode_uk_postcode(pc)
        if coords is None and city:
            coords = geocode_city(city)

        if coords:
            out.at[i, lat_col] = coords[0]
            out.at[i, lon_col] = coords[1]

    return out


# -----------------------------
# 7) Export CSV + PDF
# -----------------------------
def export_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Requirement: "Ability to export ... into ... CSV formats" :contentReference[oaicite:8]{index=8}
    """
    return df.to_csv(index=False).encode("utf-8")


def export_pdf_bytes(
    filtered_df: pd.DataFrame,
    top_df: pd.DataFrame,
    title: str = "Seizure Data Visualisation Report",
    notes: str = "",
    max_preview_rows: int = 30,
) -> bytes:
    """
    Requirement: "Ability to export the visualisations / data into PDF" :contentReference[oaicite:9]{index=9}
    Backend creates report content (summary + table preview).
    """
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = A4[1] - 2 * cm

    def draw_line(text: str, bold: bool = False, size: int = 10) -> None:
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(2 * cm, y, text[:120])
        y -= 0.55 * cm
        if y < 2.0 * cm:
            c.showPage()
            y = A4[1] - 2 * cm

    def draw_table(df: pd.DataFrame, heading: str, max_rows: int = 25) -> None:
        nonlocal y
        draw_line(heading, bold=True, size=12)
        if df.empty:
            draw_line("(No rows)")
            return

        df2 = df.head(max_rows).copy().reset_index(drop=True)
        cols = list(df2.columns)
        # rough widths
        page_w = A4[0] - 4 * cm
        col_w = max(page_w / max(len(cols), 1), 2.5 * cm)

        # header
        c.setFont("Helvetica-Bold", 9)
        x0 = 2 * cm
        for i, col in enumerate(cols):
            c.drawString(x0 + i * col_w, y, str(col)[:22])
        y -= 0.5 * cm

        # rows
        c.setFont("Helvetica", 9)
        for _, row in df2.iterrows():
            for i, col in enumerate(cols):
                v = row[col]
                s = "" if pd.isna(v) else str(v)
                c.drawString(x0 + i * col_w, y, s[:22])
            y -= 0.45 * cm
            if y < 2.0 * cm:
                c.showPage()
                y = A4[1] - 2 * cm

        y -= 0.4 * cm

    # Title + meta
    draw_line(title, bold=True, size=16)
    draw_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if notes.strip():
        draw_line(f"Notes: {notes.strip()}")

    draw_line(f"Filtered rows: {len(filtered_df):,}")

    # Top 10
    draw_table(top_df, "Top 10 Summary (filtered)", max_rows=10)

    # Data preview
    preview = filtered_df.copy()
    if len(preview.columns) > 8:
        preview = preview[preview.columns[:8]]
    draw_table(preview, "Filtered Data Preview (first rows)", max_rows=max_preview_rows)

    c.save()
    pdf = buf.getvalue()
    buf.close()
    return pdf


# -----------------------------
# 8) One “pipeline” function (backend call)
# -----------------------------
@dataclass
class BackendConfig:
    date_col: Optional[str] = None
    category_col: Optional[str] = None
    subcategory_col: Optional[str] = None
    postcode_col: Optional[str] = None
    city_col: Optional[str] = None

    # date filter
    date_mode: str = "none"
    year: Optional[int] = None
    month: Optional[int] = None
    quarter: Optional[int] = None
    start_date: Optional[Union[str, date, datetime]] = None
    end_date: Optional[Union[str, date, datetime]] = None

    # category filters
    categories: Optional[List[str]] = None
    subcategories: Optional[List[str]] = None

    # location filters
    postcodes: Optional[List[str]] = None
    cities: Optional[List[str]] = None

    # keyword
    keyword: str = ""

    # top summary
    top_column: Optional[str] = None
    top_n: int = 10


def run_backend_pipeline(df: pd.DataFrame, cfg: BackendConfig) -> Dict[str, object]:
    """
    Returns everything backend needs to give frontend:
    - filtered_df
    - geo_df (with lat/lon for heatmap)
    - top_df
    - csv_bytes
    - pdf_bytes
    """
    work = df.copy()

    # date parsing if date_col provided
    if cfg.date_col and cfg.date_col in work.columns:
        work = parse_date_column(work, cfg.date_col)

    # keyword
    work = keyword_search_filter(work, cfg.keyword)

    # date filter
    if cfg.date_col and cfg.date_col in work.columns:
        work = filter_by_date(
            work,
            date_col=cfg.date_col,
            mode=cfg.date_mode,
            year=cfg.year,
            month=cfg.month,
            quarter=cfg.quarter,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
        )

    # category/subcategory
    work = filter_by_category(
        work,
        category_col=cfg.category_col,
        subcategory_col=cfg.subcategory_col,
        categories=cfg.categories,
        subcategories=cfg.subcategories,
    )

    # location
    work = filter_by_location(
        work,
        postcode_col=cfg.postcode_col,
        city_col=cfg.city_col,
        postcodes=cfg.postcodes,
        cities=cfg.cities,
    )

    # top N
    top_df = pd.DataFrame()
    if cfg.top_column and cfg.top_column in work.columns:
        top_df = top_n_summary(work, cfg.top_column, n=cfg.top_n)

    # geocode (for heatmap)
    geo_df = attach_lat_lon(work, postcode_col=cfg.postcode_col, city_col=cfg.city_col)

    # exports
    csv_bytes = export_csv_bytes(work)
    pdf_bytes = export_pdf_bytes(work, top_df, notes=f"Keyword='{cfg.keyword}'")

    return {
        "filtered_df": work,
        "geo_df": geo_df,
        "top_df": top_df,
        "csv_bytes": csv_bytes,
        "pdf_bytes": pdf_bytes,
    }
if __name__ == "__main__":
    #  IMPORTANT: file.xlsx ka actual data "Data"
    df = pd.read_excel("/Users/d.luffy/Documents/heat-map/file.xlsx", sheet_name="Data")

    cfg = BackendConfig(
        date_col="Date of Seizure",
        category_col="Category",
        subcategory_col="Commodity",
        postcode_col="Postcode",
        city_col=None,                 
        date_mode="yearly",
        year=2025,                     
        keyword="",                    

        top_column="Commodity",       
        top_n=10
    )

    result = run_backend_pipeline(df, cfg)

    print(" Filtered rows:", len(result["filtered_df"]))
    print("\n Top 10 Summary:\n", result["top_df"].head(10))

    # exports (save to Desktop folder)
import os

out_dir = os.path.expanduser("~/Desktop/heatmap_outputs")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "filtered_output.csv"), "wb") as f:
    f.write(result["csv_bytes"])

with open(os.path.join(out_dir, "report.pdf"), "wb") as f:
    f.write(result["pdf_bytes"])

# geo_df heatmap
result["geo_df"].to_csv(os.path.join(out_dir, "geo_output.csv"), index=False)

print("\n Files saved in:", out_dir)