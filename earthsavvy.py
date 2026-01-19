# code.py
"""
Thermal-to-CAPEX (EarthSavvy Add-on) | Offline Streamlit Tool
============================================================

How it works (screening-grade)
- You upload (or select from the local folder) EarthSavvy JSON exports and any optional supporting files (CSV/GeoJSON).
- The app defensively parses all inputs into one canonical "long" table:
    timestamp (optional), location_id/name, metric_name, metric_value, units, geometry (optional), source_file
- You select a temperature / anomaly signal and baseline approach.
- A simplified physics model estimates outward heat flux (convection + longwave radiation), then annualises to kWh.
- Annual cost (£) and carbon (tCO2e) are computed using user-supplied tariff and carbon factors.
- Uncertainty bounds are estimated with Monte Carlo sampling (low/central/high).
- Buildings are ranked and labelled using an explainable ruleset (Fabric loss vs Process hotspot vs Mixed/Uncertain).
- Results can be exported to CSV, JSON, and a simple offline HTML report.

Notes
- This is a screening tool designed for rapid CAPEX justification. It is not a calibrated heat-balance model.
- It runs offline and avoids external services. Maps are shown with a blank background by default (no internet tiles).
"""

from __future__ import annotations

import os
import io
import json
import math
import re
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------
# Constants
# -----------------------------
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann, W·m−2·K−4

LIKELY_TEMP_KEYWORDS = [
    "temp", "temperature", "thermal", "hotspot", "anomaly", "delta", "deltat", "dt", "surface"
]
LIKELY_AMBIENT_KEYWORDS = ["ambient", "air_temp", "outside", "external", "background", "baseline"]
LIKELY_AREA_KEYWORDS = ["area", "m2", "sqm", "footprint", "roof_area", "floor_area", "surface_area"]

DEFAULT_REQ_TXT = """streamlit==1.37.1
pandas==2.2.2
numpy==2.0.1
plotly==5.23.0
"""

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Thermal-to-CAPEX (EarthSavvy Add-on)",
    layout="wide",
)

st.title("Thermal-to-CAPEX (EarthSavvy Add-on)")
st.caption("Offline screening tool: thermal evidence → heat-loss estimate → annual kWh, £ and tCO₂e with uncertainty.")


# -----------------------------
# Utility helpers
# -----------------------------
def _safe_lower(x: Any) -> str:
    try:
        return str(x).strip().lower()
    except Exception:
        return ""


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not (isinstance(x, bool))


def _try_parse_datetime(x: Any) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x, errors="coerce")
    s = str(x).strip()
    if not s:
        return None
    # Common EarthSavvy-like formats:
    # "2024-11-01 11:12", "2024-11-01T11:12:00Z", etc.
    ts = pd.to_datetime(s, errors="coerce", utc=False)
    if pd.isna(ts):
        return None
    return ts


def _normalise_units(u: Any) -> Optional[str]:
    if u is None:
        return None
    s = str(u).strip()
    if not s:
        return None
    # A few common normalisations
    s = s.replace("°C", "C").replace("degC", "C").replace("℃", "C")
    s = s.replace("W/m2", "W/m²").replace("W/m^2", "W/m²")
    return s


def list_local_files(extensions: Tuple[str, ...] = (".json", ".csv", ".geojson", ".pdf", ".png", ".jpg", ".jpeg")) -> List[str]:
    cwd = os.getcwd()
    files = []
    for fn in os.listdir(cwd):
        p = os.path.join(cwd, fn)
        if not os.path.isfile(p):
            continue
        if fn.lower().endswith(extensions):
            files.append(fn)
    return sorted(files)


def _read_local_file_bytes(filename: str) -> bytes:
    with open(filename, "rb") as f:
        return f.read()


def _detect_file_type(filename: str) -> str:
    fn = filename.lower()
    if fn.endswith(".geojson"):
        return "geojson"
    if fn.endswith(".json"):
        return "json"
    if fn.endswith(".csv"):
        return "csv"
    if fn.endswith(".pdf"):
        return "pdf"
    if fn.endswith((".png", ".jpg", ".jpeg")):
        return "image"
    return "unknown"


def _preview_df(df: pd.DataFrame, n: int = 50) -> None:
    if df is None or df.empty:
        st.info("No rows to preview.")
        return
    st.dataframe(df.head(n), use_container_width=True)


def _guess_id_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    candidates = []
    for c in df.columns:
        cl = _safe_lower(c)
        if any(k in cl for k in ["building", "site", "location", "id", "name"]):
            candidates.append(c)
    if candidates:
        # Prefer most explicit "id"/"name"
        for k in ["location_id", "building_id", "site_id", "id", "name", "building", "location", "site"]:
            for c in candidates:
                if k in _safe_lower(c):
                    return c
        return candidates[0]
    return df.columns[0]


def _guess_area_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for c in df.columns:
        cl = _safe_lower(c)
        if any(k in cl for k in LIKELY_AREA_KEYWORDS):
            return c
    return None


def _extract_lat_lon_from_props(props: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat = None
    lon = None
    for k, v in props.items():
        kl = _safe_lower(k)
        if lat is None and kl in ["lat", "latitude", "y"]:
            if _is_number(v):
                lat = float(v)
        if lon is None and kl in ["lon", "lng", "longitude", "x"]:
            if _is_number(v):
                lon = float(v)
    return lat, lon


def _centroid_of_polygon_lonlat(coords: List[List[float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Very lightweight centroid for lon/lat polygon ring (not projected, screening only).
    coords: list of [lon, lat]
    """
    if not coords or len(coords) < 3:
        return None, None
    xs = [c[0] for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
    ys = [c[1] for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
    if not xs or not ys:
        return None, None
    return float(np.mean(xs)), float(np.mean(ys))


def _extract_lonlat_from_geometry(geom: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(geom, dict):
        return None, None
    gtype = _safe_lower(geom.get("type"))
    coords = geom.get("coordinates")
    if gtype == "point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return float(coords[0]), float(coords[1])
    if gtype == "polygon" and isinstance(coords, list) and coords:
        ring = coords[0]
        if isinstance(ring, list) and ring:
            return _centroid_of_polygon_lonlat(ring)
    if gtype == "multipolygon" and isinstance(coords, list) and coords:
        # Take centroid of first polygon's first ring
        try:
            ring = coords[0][0]
            return _centroid_of_polygon_lonlat(ring)
        except Exception:
            return None, None
    return None, None


# -----------------------------
# JSON parsing and canonicalisation
# -----------------------------
def _context_update(context: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
    """
    Update context with helpful fields if key/value looks like time, location, units, etc.
    """
    kl = _safe_lower(key)

    # Timestamp hints
    if any(tk in kl for tk in ["timestamp", "datetime", "date_time", "time", "date"]):
        ts = _try_parse_datetime(value)
        if ts is not None:
            context["timestamp"] = ts

    # Units hints
    if kl in ["unit", "units", "uom"]:
        u = _normalise_units(value)
        if u:
            context["units"] = u

    # Location hints
    if any(lk in kl for lk in ["location", "site", "building", "asset", "name", "id"]):
        # Keep the most specific non-empty value
        if isinstance(value, (str, int)) and str(value).strip():
            # Prefer explicit ids/names
            if "location_id" not in context and ("id" in kl or "location" in kl or "building" in kl or "site" in kl):
                context["location_id"] = str(value).strip()
            if "location_name" not in context and ("name" in kl or "title" in kl or "label" in kl):
                context["location_name"] = str(value).strip()

    # Lat/lon hints
    if kl in ["lat", "latitude", "y"]:
        if _is_number(value):
            context["lat"] = float(value)
    if kl in ["lon", "lng", "longitude", "x"]:
        if _is_number(value):
            context["lon"] = float(value)

    # Geometry hints (GeoJSON-like)
    if kl == "geometry" and isinstance(value, dict):
        context["geometry"] = value
        lon, lat = _extract_lonlat_from_geometry(value)
        if lon is not None and lat is not None:
            context["lon"] = lon
            context["lat"] = lat

    # Area hints
    if any(k in kl for k in LIKELY_AREA_KEYWORDS) and _is_number(value):
        context["area_m2"] = float(value)

    return context


def _is_timeseries_table(obj: Any) -> bool:
    """
    Detect a list-of-lists structure with first column parseable as datetime and remaining numeric.
    """
    if not isinstance(obj, list) or len(obj) < 2:
        return False
    # Must be list of lists/tuples
    if not all(isinstance(r, (list, tuple)) for r in obj):
        return False
    # Check a few rows
    sample = obj[: min(5, len(obj))]
    for r in sample:
        if len(r) < 2:
            return False
        ts = _try_parse_datetime(r[0])
        if ts is None:
            return False
        # At least one numeric after timestamp
        if not any(_is_number(x) for x in r[1:]):
            return False
    return True


def _parse_timeseries_table(
    rows: List[List[Any]],
    parent_metric_prefix: str,
    context: Dict[str, Any],
    source_file: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    # Determine number of columns
    ncols = max(len(r) for r in rows) if rows else 0
    metric_names = [f"{parent_metric_prefix}_col{i}" for i in range(1, ncols)]
    for r in rows:
        if len(r) < 2:
            continue
        ts = _try_parse_datetime(r[0])
        if ts is None:
            continue
        for j in range(1, len(r)):
            v = r[j]
            if not _is_number(v):
                continue
            rec = {
                "timestamp": ts,
                "location_id": context.get("location_id"),
                "location_name": context.get("location_name"),
                "metric_name": metric_names[j - 1],
                "metric_value": float(v),
                "units": context.get("units"),
                "geometry": context.get("geometry"),
                "lat": context.get("lat"),
                "lon": context.get("lon"),
                "area_m2": context.get("area_m2"),
                "source_file": source_file,
            }
            records.append(rec)
    return records


def _walk_json_to_records(
    obj: Any,
    path: str,
    context: Dict[str, Any],
    source_file: str,
    max_records: int = 250000,
) -> List[Dict[str, Any]]:
    """
    Walk arbitrary JSON and extract numeric observations into canonical record dicts.
    This is intentionally defensive and schema-agnostic.
    """
    records: List[Dict[str, Any]] = []

    def _walk(o: Any, p: str, ctx: Dict[str, Any]) -> None:
        if len(records) >= max_records:
            return

        # Timeseries list-of-lists pattern
        if _is_timeseries_table(o):
            records.extend(_parse_timeseries_table(o, p, ctx, source_file))
            return

        if isinstance(o, dict):
            # First pass: update context from keys
            new_ctx = dict(ctx)
            for k, v in o.items():
                new_ctx = _context_update(new_ctx, k, v)

            # Second pass: recurse and also handle leaf numerics
            for k, v in o.items():
                kp = f"{p}.{k}" if p else str(k)

                if _is_number(v):
                    rec = {
                        "timestamp": new_ctx.get("timestamp"),
                        "location_id": new_ctx.get("location_id"),
                        "location_name": new_ctx.get("location_name"),
                        "metric_name": kp,
                        "metric_value": float(v),
                        "units": new_ctx.get("units"),
                        "geometry": new_ctx.get("geometry"),
                        "lat": new_ctx.get("lat"),
                        "lon": new_ctx.get("lon"),
                        "area_m2": new_ctx.get("area_m2"),
                        "source_file": source_file,
                    }
                    records.append(rec)
                    continue

                if isinstance(v, (dict, list)):
                    _walk(v, kp, new_ctx)
                    continue

            return

        if isinstance(o, list):
            # If list of dicts, recurse for each item
            for idx, item in enumerate(o):
                ip = f"{p}[{idx}]"
                if isinstance(item, (dict, list)):
                    _walk(item, ip, dict(ctx))
                elif _is_number(item):
                    rec = {
                        "timestamp": ctx.get("timestamp"),
                        "location_id": ctx.get("location_id"),
                        "location_name": ctx.get("location_name"),
                        "metric_name": ip,
                        "metric_value": float(item),
                        "units": ctx.get("units"),
                        "geometry": ctx.get("geometry"),
                        "lat": ctx.get("lat"),
                        "lon": ctx.get("lon"),
                        "area_m2": ctx.get("area_m2"),
                        "source_file": source_file,
                    }
                    records.append(rec)
            return

        # Leaf primitives ignored (strings/bools etc)

    _walk(obj, path, context)
    return records


@st.cache_data(show_spinner=False)
def parse_json_bytes(file_bytes: bytes, source_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      df_long: canonical long-format dataframe
      meta: parsed metadata useful for preview
    """
    meta: Dict[str, Any] = {"source_file": source_name}
    try:
        obj = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        return pd.DataFrame(), {"source_file": source_name, "error": f"Could not parse JSON: {e}"}

    # Preview metadata (top-level keys)
    if isinstance(obj, dict):
        meta["top_level_keys"] = list(obj.keys())[:200]
    else:
        meta["top_level_type"] = type(obj).__name__

    records = _walk_json_to_records(obj, path="", context={}, source_file=source_name)
    if not records:
        return pd.DataFrame(), {**meta, "warning": "No numeric metrics were found in this JSON."}

    df = pd.DataFrame.from_records(records)
    # Normalise timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Fill location fallback
    if "location_id" in df.columns:
        df["location_id"] = df["location_id"].fillna("")
    if "location_name" in df.columns:
        df["location_name"] = df["location_name"].fillna("")

    # Build a robust location label for UI
    df["location_label"] = df.apply(
        lambda r: (r.get("location_name") or "").strip() or (r.get("location_id") or "").strip() or "Unspecified location",
        axis=1,
    )

    # Units cleanup
    if "units" in df.columns:
        df["units"] = df["units"].apply(_normalise_units)

    return df, meta


@st.cache_data(show_spinner=False)
def parse_csv_bytes(file_bytes: bytes, source_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"source_file": source_name}
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
        except Exception as e:
            return pd.DataFrame(), {"source_file": source_name, "error": f"Could not parse CSV: {e}"}

    meta["columns"] = list(df.columns)
    meta["rows"] = int(df.shape[0])

    # Try detect timestamp columns for preview
    ts_col = None
    for c in df.columns:
        cl = _safe_lower(c)
        if "time" in cl or "date" in cl:
            ts_col = c
            break
    if ts_col:
        meta["timestamp_column_guess"] = ts_col

    return df, meta


@st.cache_data(show_spinner=False)
def parse_geojson_bytes(file_bytes: bytes, source_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Minimal GeoJSON parsing without geopandas/shapely.
    Outputs a feature table with lat/lon centroid when possible.
    """
    meta: Dict[str, Any] = {"source_file": source_name}
    try:
        obj = json.loads(file_bytes.decode("utf-8", errors="replace"))
    except Exception as e:
        return pd.DataFrame(), {"source_file": source_name, "error": f"Could not parse GeoJSON: {e}"}

    features = obj.get("features") if isinstance(obj, dict) else None
    if not isinstance(features, list):
        return pd.DataFrame(), {**meta, "error": "GeoJSON does not contain a 'features' list."}

    rows = []
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}

        lon, lat = _extract_lonlat_from_geometry(geom)
        if lon is None or lat is None:
            # Try from properties
            lat2, lon2 = _extract_lat_lon_from_props(props)
            lat = lat if lat is not None else lat2
            lon = lon if lon is not None else lon2

        # Build ID / name
        fid = feat.get("id")
        name = None
        for k in ["name", "building", "site", "location", "asset", "id"]:
            if k in props and props.get(k) is not None:
                name = str(props.get(k)).strip()
                break

        # Area guess
        area_m2 = None
        for k, v in props.items():
            if any(kw in _safe_lower(k) for kw in LIKELY_AREA_KEYWORDS) and _is_number(v):
                area_m2 = float(v)
                break

        rows.append(
            {
                "feature_index": i,
                "feature_id": str(fid).strip() if fid is not None else "",
                "feature_name": name or "",
                "lat": lat,
                "lon": lon,
                "area_m2": area_m2,
                "properties": props,
                "geometry": geom,
                "source_file": source_name,
            }
        )

    df = pd.DataFrame(rows)
    meta["features"] = int(df.shape[0])
    meta["columns"] = list(df.columns)
    return df, meta


def build_canonical_long_table(
    json_long_tables: List[pd.DataFrame],
) -> pd.DataFrame:
    if not json_long_tables:
        return pd.DataFrame(columns=[
            "timestamp", "location_label", "metric_name", "metric_value", "units", "geometry", "lat", "lon", "area_m2", "source_file"
        ])
    df = pd.concat(json_long_tables, ignore_index=True)
    keep_cols = ["timestamp", "location_label", "location_id", "location_name",
                 "metric_name", "metric_value", "units", "geometry", "lat", "lon", "area_m2", "source_file"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep_cols].copy()
    return df


# -----------------------------
# Heat-loss and CAPEX estimation
# -----------------------------
def _pick_metric_candidates(metric_names: List[str]) -> Tuple[List[str], List[str]]:
    """
    Returns (temp_like, ambient_like)
    """
    temp_like = []
    ambient_like = []
    for m in metric_names:
        ml = _safe_lower(m)
        if any(k in ml for k in LIKELY_TEMP_KEYWORDS):
            temp_like.append(m)
        if any(k in ml for k in LIKELY_AMBIENT_KEYWORDS):
            ambient_like.append(m)
    return temp_like, ambient_like


def _compute_heat_flux_wm2(delta_t_c: np.ndarray, t_amb_c: np.ndarray, h: float, emissivity: float) -> np.ndarray:
    """
    delta_t_c: Ts - Ta in °C (same as K difference)
    t_amb_c: ambient air temperature °C
    """
    # Convection
    q_conv = h * np.clip(delta_t_c, 0.0, None)

    # Longwave radiation (approx; assumes surface and ambient emit to each other)
    ta_k = (t_amb_c + 273.15).astype(float)
    ts_k = (t_amb_c + delta_t_c + 273.15).astype(float)
    q_rad = emissivity * SIGMA_SB * (np.power(ts_k, 4) - np.power(ta_k, 4))

    q_total = q_conv + q_rad
    q_total = np.clip(q_total, 0.0, None)
    return q_total


def _monte_carlo_bounds(
    delta_t_stat: float,
    t_amb_c: float,
    area_m2: float,
    tariff_gbp_per_kwh: float,
    carbon_kg_per_kwh: float,
    h_central: float,
    h_min: float,
    h_max: float,
    eps_central: float,
    eps_min: float,
    eps_max: float,
    seasonal_central: float,
    seasonal_min: float,
    seasonal_max: float,
    area_unc_frac: float,
    amb_unc_c: float,
    tariff_unc_frac: float,
    carbon_unc_frac: float,
    n: int,
    agg_method: str,
) -> Dict[str, float]:
    """
    Sampling-based bounds for annual energy/cost/carbon.
    """
    n = int(max(50, min(5000, n)))
    rng = np.random.default_rng(42)

    hs = rng.uniform(h_min, h_max, n)
    eps = rng.uniform(eps_min, eps_max, n)

    seasonal = rng.uniform(seasonal_min, seasonal_max, n)
    # Area uncertainty as multiplicative factor
    area_factor = rng.uniform(max(0.05, 1.0 - area_unc_frac), 1.0 + area_unc_frac, n)
    # Ambient uncertainty affects radiation term
    amb = rng.uniform(t_amb_c - abs(amb_unc_c), t_amb_c + abs(amb_unc_c), n)

    tariff = rng.uniform(max(0.0, tariff_gbp_per_kwh * (1.0 - tariff_unc_frac)),
                         tariff_gbp_per_kwh * (1.0 + tariff_unc_frac), n)
    carbon = rng.uniform(max(0.0, carbon_kg_per_kwh * (1.0 - carbon_unc_frac)),
                         carbon_kg_per_kwh * (1.0 + carbon_unc_frac), n)

    # If aggregation is peak-ish, we allow deltaT uncertainty widening a bit
    dt = np.full(n, float(delta_t_stat))
    if agg_method in ["90th percentile ΔT", "Peak ΔT"]:
        dt = dt * rng.uniform(0.9, 1.15, n)

    q_wm2 = _compute_heat_flux_wm2(dt, amb, hs, eps)
    power_w = q_wm2 * (area_m2 * area_factor)
    annual_kwh = power_w * 8760.0 * seasonal / 1000.0

    annual_cost = annual_kwh * tariff
    annual_tco2 = annual_kwh * carbon / 1000.0  # kg→t

    def p(x: np.ndarray, q: float) -> float:
        return float(np.percentile(x, q))

    return {
        "annual_kwh_p10": p(annual_kwh, 10),
        "annual_kwh_p50": p(annual_kwh, 50),
        "annual_kwh_p90": p(annual_kwh, 90),
        "annual_cost_p10": p(annual_cost, 10),
        "annual_cost_p50": p(annual_cost, 50),
        "annual_cost_p90": p(annual_cost, 90),
        "annual_tco2_p10": p(annual_tco2, 10),
        "annual_tco2_p50": p(annual_tco2, 50),
        "annual_tco2_p90": p(annual_tco2, 90),
    }


def _classify_hotspot(
    dt_median: float,
    dt_cv_time: Optional[float],
    spatial_cv: Optional[float],
    max_dt: Optional[float],
) -> str:
    """
    Explainable screening rules:
    - Fabric losses tend to be persistent and moderately uniform.
    - Process hotspots tend to be intermittent / highly variable and/or spatially concentrated.
    """
    dt_cv_time = float(dt_cv_time) if dt_cv_time is not None and np.isfinite(dt_cv_time) else None
    spatial_cv = float(spatial_cv) if spatial_cv is not None and np.isfinite(spatial_cv) else None
    max_dt = float(max_dt) if max_dt is not None and np.isfinite(max_dt) else None

    # Insufficient evidence
    if dt_median is None or not np.isfinite(dt_median):
        return "Uncertain"

    # Primary rules
    likely_fabric = (dt_median >= 3.0) and ((dt_cv_time is None) or (dt_cv_time < 0.35)) and ((spatial_cv is None) or (spatial_cv < 0.40))
    likely_process = ((dt_cv_time is not None and dt_cv_time > 0.60) or (spatial_cv is not None and spatial_cv > 0.70))
    if max_dt is not None:
        likely_process = likely_process or (max_dt >= 15.0 and dt_median < 6.0)

    if likely_fabric and not likely_process:
        return "Likely Fabric Loss"
    if likely_process and not likely_fabric:
        return "Likely Process Hotspot"
    if likely_fabric and likely_process:
        return "Mixed/Uncertain"
    return "Mixed/Uncertain"


# -----------------------------
# Session state
# -----------------------------
if "loaded_sources" not in st.session_state:
    st.session_state.loaded_sources = {}  # source_name -> dict(type, df/meta)
if "canonical_long" not in st.session_state:
    st.session_state.canonical_long = pd.DataFrame()
if "geo_features" not in st.session_state:
    st.session_state.geo_features = pd.DataFrame()
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = pd.DataFrame()
if "analysis_notes" not in st.session_state:
    st.session_state.analysis_notes = []


# -----------------------------
# Tabs
# -----------------------------
tab_upload, tab_settings, tab_results, tab_export = st.tabs(
    ["Upload & Detect", "Analysis Settings", "Results", "Export"]
)

with tab_upload:
    st.subheader("Upload & Detect")

    colA, colB = st.columns([1.2, 1.0], gap="large")

    with colA:
        st.markdown("#### Upload files (optional)")
        uploaded = st.file_uploader(
            "Upload one or more files (JSON / CSV / GeoJSON).",
            accept_multiple_files=True,
            type=["json", "csv", "geojson", "pdf", "png", "jpg", "jpeg"],
        )

        st.markdown("#### Or select from current folder")
        local_files = list_local_files()
        selected_local = st.multiselect(
            "Files found in the working directory:",
            options=local_files,
            default=[],
        )

        load_clicked = st.button("Load selected files", type="primary")

    with colB:
        st.markdown("#### Loaded sources")
        if not st.session_state.loaded_sources:
            st.info("No sources loaded yet.")
        else:
            for src, info in st.session_state.loaded_sources.items():
                st.write(f"- **{src}** ({info.get('type', 'unknown')})")

    def _load_source(name: str, ftype: str, file_bytes: bytes) -> None:
        if ftype == "json":
            df, meta = parse_json_bytes(file_bytes, name)
            st.session_state.loaded_sources[name] = {"type": "json", "df": df, "meta": meta}
        elif ftype == "csv":
            df, meta = parse_csv_bytes(file_bytes, name)
            st.session_state.loaded_sources[name] = {"type": "csv", "df": df, "meta": meta}
        elif ftype == "geojson":
            df, meta = parse_geojson_bytes(file_bytes, name)
            st.session_state.loaded_sources[name] = {"type": "geojson", "df": df, "meta": meta}
        else:
            # Keep as raw bytes for context files (pdf/images)
            st.session_state.loaded_sources[name] = {"type": ftype, "bytes": file_bytes, "meta": {"source_file": name}}

    # Load uploaded files immediately
    if uploaded:
        for uf in uploaded:
            name = uf.name
            ftype = _detect_file_type(name)
            b = uf.getvalue()
            # Avoid reloading identical content
            key = f"{name}::{_hash_bytes(b)}"
            # Keep user-visible name stable even if reuploaded
            _load_source(name, ftype, b)
        st.success(f"Loaded {len(uploaded)} uploaded file(s).")

    # Load selected local files on click
    if load_clicked and selected_local:
        for fn in selected_local:
            b = _read_local_file_bytes(fn)
            ftype = _detect_file_type(fn)
            _load_source(fn, ftype, b)
        st.success(f"Loaded {len(selected_local)} local file(s).")

    # Build canonical long table from all parsed JSON sources
    json_tables = []
    geo_tables = []
    csv_tables = []
    for src, info in st.session_state.loaded_sources.items():
        if info.get("type") == "json" and isinstance(info.get("df"), pd.DataFrame) and not info["df"].empty:
            json_tables.append(info["df"])
        if info.get("type") == "geojson" and isinstance(info.get("df"), pd.DataFrame) and not info["df"].empty:
            geo_tables.append(info["df"])
        if info.get("type") == "csv" and isinstance(info.get("df"), pd.DataFrame) and not info["df"].empty:
            csv_tables.append(info["df"])

    st.session_state.canonical_long = build_canonical_long_table(json_tables)
    st.session_state.geo_features = pd.concat(geo_tables, ignore_index=True) if geo_tables else pd.DataFrame()

    st.markdown("---")

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("#### Canonicalised metrics (from JSON)")
        df_long = st.session_state.canonical_long
        if df_long.empty:
            st.warning("No numeric metrics detected yet. Upload at least one JSON export with numeric values.")
        else:
            st.write(
                f"Rows: **{len(df_long):,}** | "
                f"Locations: **{df_long['location_label'].nunique():,}** | "
                f"Metrics: **{df_long['metric_name'].nunique():,}**"
            )
            _preview_df(df_long[["timestamp", "location_label", "metric_name", "metric_value", "units", "source_file"]], n=40)

            with st.expander("Inferred schema details"):
                st.write("Columns in canonical table:")
                st.code("\n".join(df_long.columns), language="text")
                st.write("Example metric names:")
                example_metrics = sorted(df_long["metric_name"].dropna().unique().tolist())[:50]
                st.code("\n".join(example_metrics), language="text")

    with right:
        st.markdown("#### Optional geometry / footprints (GeoJSON)")
        geo_df = st.session_state.geo_features
        if geo_df.empty:
            st.info("No GeoJSON features loaded. If you upload GeoJSON building footprints, the app can attach lat/lon and area metadata where available.")
        else:
            st.write(f"Features: **{len(geo_df):,}**")
            _preview_df(geo_df[["feature_id", "feature_name", "lat", "lon", "area_m2", "source_file"]], n=25)

        st.markdown("#### Other CSVs")
        if not csv_tables:
            st.info("No CSVs loaded.")
        else:
            for i, cdf in enumerate(csv_tables[:3], start=1):
                st.write(f"CSV #{i}: {cdf.shape[0]:,} rows × {cdf.shape[1]:,} cols")
                _preview_df(cdf, n=15)


with tab_settings:
    st.subheader("Analysis Settings")

    df_long = st.session_state.canonical_long

    if df_long.empty:
        st.warning("Upload and detect data first.")
    else:
        metric_names = sorted(df_long["metric_name"].dropna().unique().tolist())
        temp_like, ambient_like = _pick_metric_candidates(metric_names)

        col1, col2, col3 = st.columns([1.1, 1.0, 1.0], gap="large")

        with col1:
            st.markdown("#### Signal selection")
            auto_metric = None
            for m in temp_like:
                ml = _safe_lower(m)
                # Prefer anomaly-like metrics first
                if "anomaly" in ml or "delta" in ml or "dt" in ml:
                    auto_metric = m
                    break
            if auto_metric is None and temp_like:
                auto_metric = temp_like[0]
            if auto_metric is None and metric_names:
                auto_metric = metric_names[0]

            selected_metric = st.selectbox(
                "Metric to interpret as temperature signal (surface temperature or anomaly):",
                options=metric_names,
                index=metric_names.index(auto_metric) if auto_metric in metric_names else 0,
            )

            metric_is_deltaT = st.checkbox(
                "This metric already represents ΔT (surface minus ambient) in °C/K",
                value=("anomaly" in _safe_lower(selected_metric) or "delta" in _safe_lower(selected_metric) or re.search(r"\bdt\b", _safe_lower(selected_metric)) is not None),
            )

            st.markdown("#### Baseline / ambient")
            baseline_method = st.selectbox(
                "Ambient baseline method (used when metric is surface temperature, or for radiation term):",
                options=[
                    "Constant ambient temperature",
                    "Percentile baseline per building (10th percentile)",
                    "Use ambient metric from data (if available)",
                ],
                index=0,
            )

            ambient_metric = None
            if ambient_like:
                ambient_metric = ambient_like[0]
            ambient_metric = st.selectbox(
                "Ambient metric (only used if baseline method is 'Use ambient metric from data'):",
                options=["(none)"] + ambient_like,
                index=(1 if ambient_like else 0),
            )

            ambient_constant_c = st.number_input(
                "Constant ambient temperature (°C)",
                value=10.0,
                step=0.5,
            )

            st.markdown("#### Time aggregation")
            agg_method = st.selectbox(
                "Aggregation approach for ΔT over time:",
                options=["Mean positive ΔT", "Median positive ΔT", "90th percentile ΔT", "Peak ΔT"],
                index=0,
            )

        with col2:
            st.markdown("#### Geometry and area")
            default_area_m2 = st.number_input(
                "Default effective area per building (m²) if unknown",
                value=500.0,
                step=10.0,
                help="Used when footprint/area cannot be inferred. Screening value only.",
            )

            use_geo_join = st.checkbox(
                "Try to attach GeoJSON feature area and lat/lon to locations (best-effort match on name/id)",
                value=True,
            )

            # Uncertainty settings
            st.markdown("#### Uncertainty ranges")
            area_unc_frac = st.slider(
                "Area uncertainty (± fraction)",
                min_value=0.0,
                max_value=1.0,
                value=0.30,
                step=0.05,
                help="Represents uncertainty in effective heat-loss area (roof/walls exposed, sensor footprint, etc.).",
            )
            amb_unc_c = st.slider(
                "Ambient temperature uncertainty (± °C)",
                min_value=0.0,
                max_value=15.0,
                value=3.0,
                step=0.5,
            )

            st.markdown("#### Scaling to annual")
            seasonal_central = st.slider(
                "Seasonal / operating fraction (central)",
                min_value=0.05,
                max_value=1.0,
                value=0.50,
                step=0.05,
                help="Fraction of the year the measured ΔT is assumed representative (heating season, operating hours, etc.).",
            )
            seasonal_min = st.slider("Seasonal fraction (min)", 0.05, 1.0, 0.35, 0.05)
            seasonal_max = st.slider("Seasonal fraction (max)", 0.05, 1.0, 0.70, 0.05)

        with col3:
            st.markdown("#### Physics parameters")
            h_central = st.number_input(
                "Convective heat transfer coefficient h (W/m²·K) central",
                value=10.0,
                step=0.5,
            )
            h_min = st.number_input("h min (W/m²·K)", value=5.0, step=0.5)
            h_max = st.number_input("h max (W/m²·K)", value=25.0, step=0.5)

            eps_central = st.slider("Emissivity ε (central)", 0.30, 0.98, 0.90, 0.01)
            eps_min = st.slider("Emissivity ε (min)", 0.30, 0.98, 0.80, 0.01)
            eps_max = st.slider("Emissivity ε (max)", 0.30, 0.98, 0.95, 0.01)

            st.markdown("#### Commercial factors")
            tariff_gbp_per_kwh = st.number_input(
                "Tariff (£/kWh)",
                value=0.18,
                step=0.01,
                help="Simple blended electricity price for screening.",
            )
            tariff_unc_frac = st.slider("Tariff uncertainty (± fraction)", 0.0, 1.0, 0.25, 0.05)

            carbon_kg_per_kwh = st.number_input(
                "Carbon factor (kgCO₂e/kWh)",
                value=0.20,
                step=0.01,
                help="Use a site-appropriate electricity factor (screening).",
            )
            carbon_unc_frac = st.slider("Carbon uncertainty (± fraction)", 0.0, 1.0, 0.30, 0.05)

            st.markdown("#### Monte Carlo")
            n_mc = st.slider("Monte Carlo samples", 50, 2000, 300, 50)

        st.markdown("---")
        st.markdown("#### Optional: per-location area overrides (editable)")
        locations = sorted(df_long["location_label"].dropna().unique().tolist())
        if len(locations) > 0:
            # Build a small editable table for area overrides
            if "area_override_df" not in st.session_state:
                st.session_state.area_override_df = pd.DataFrame({
                    "location_label": locations[: min(len(locations), 50)],
                    "area_m2_override": [np.nan] * min(len(locations), 50),
                })

            area_override_df = st.data_editor(
                st.session_state.area_override_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "location_label": st.column_config.TextColumn("Location", width="large"),
                    "area_m2_override": st.column_config.NumberColumn("Area override (m²)", min_value=0.0, step=1.0),
                },
            )
            st.session_state.area_override_df = area_override_df
        else:
            st.info("No locations available to edit.")


with tab_results:
    st.subheader("Results")

    df_long = st.session_state.canonical_long
    if df_long.empty:
        st.warning("Upload and detect data first.")
    else:
        # Pull settings from widgets in previous tab (Streamlit retains values)
        # Re-read them defensively with defaults if missing
        selected_metric = st.session_state.get("selected_metric", None)
        # We cannot directly access widget variables from other tab reliably, so infer from rerun state
        # Instead, recompute from current UI by looking up widget keys via Streamlit internals is not supported.
        # Practical approach: re-derive from dataframe and store in session each rerun using the same labels.
        # Here we re-create choices quickly in hidden manner by reading from widget state.
        # Streamlit stores widget state in session_state by label-derived keys; to keep robust, we re-extract by scanning.

        # Helper to fetch from session_state by searching keys
        def _get_state_by_partial(partial: str, default: Any) -> Any:
            for k, v in st.session_state.items():
                if partial in str(k):
                    return v
            return default

        selected_metric = _get_state_by_partial("Metric to interpret as temperature signal", None)
        if selected_metric is None:
            # Fallback
            metric_names = sorted(df_long["metric_name"].dropna().unique().tolist())
            selected_metric = metric_names[0]

        metric_is_deltaT = _get_state_by_partial("This metric already represents ΔT", False)
        baseline_method = _get_state_by_partial("Ambient baseline method", "Constant ambient temperature")
        ambient_metric = _get_state_by_partial("Ambient metric", "(none)")
        ambient_constant_c = float(_get_state_by_partial("Constant ambient temperature", 10.0))
        agg_method = _get_state_by_partial("Aggregation approach for ΔT", "Mean positive ΔT")

        default_area_m2 = float(_get_state_by_partial("Default effective area per building", 500.0))
        use_geo_join = bool(_get_state_by_partial("Try to attach GeoJSON feature area and lat/lon", True))

        area_unc_frac = float(_get_state_by_partial("Area uncertainty (± fraction)", 0.30))
        amb_unc_c = float(_get_state_by_partial("Ambient temperature uncertainty (± °C)", 3.0))

        seasonal_central = float(_get_state_by_partial("Seasonal / operating fraction (central)", 0.50))
        seasonal_min = float(_get_state_by_partial("Seasonal fraction (min)", 0.35))
        seasonal_max = float(_get_state_by_partial("Seasonal fraction (max)", 0.70))

        h_central = float(_get_state_by_partial("Convective heat transfer coefficient h", 10.0))
        h_min = float(_get_state_by_partial("h min", 5.0))
        h_max = float(_get_state_by_partial("h max", 25.0))

        eps_central = float(_get_state_by_partial("Emissivity ε (central)", 0.90))
        eps_min = float(_get_state_by_partial("Emissivity ε (min)", 0.80))
        eps_max = float(_get_state_by_partial("Emissivity ε (max)", 0.95))

        tariff_gbp_per_kwh = float(_get_state_by_partial("Tariff (£/kWh)", 0.18))
        tariff_unc_frac = float(_get_state_by_partial("Tariff uncertainty (± fraction)", 0.25))

        carbon_kg_per_kwh = float(_get_state_by_partial("Carbon factor (kgCO₂e/kWh)", 0.20))
        carbon_unc_frac = float(_get_state_by_partial("Carbon uncertainty (± fraction)", 0.30))

        n_mc = int(_get_state_by_partial("Monte Carlo samples", 300))

        # Area overrides
        area_override_df = st.session_state.get("area_override_df", pd.DataFrame(columns=["location_label", "area_m2_override"]))
        area_override_map = {}
        if isinstance(area_override_df, pd.DataFrame) and not area_override_df.empty:
            for _, r in area_override_df.iterrows():
                loc = str(r.get("location_label", "")).strip()
                val = r.get("area_m2_override", np.nan)
                if loc and _is_number(val) and float(val) > 0:
                    area_override_map[loc] = float(val)

        # Best-effort geo join
        geo_df = st.session_state.geo_features.copy()
        geo_map_area = {}
        geo_map_latlon = {}

        if use_geo_join and isinstance(geo_df, pd.DataFrame) and not geo_df.empty:
            # Create a normalised join key
            def _norm_key(s: str) -> str:
                s = _safe_lower(s)
                s = re.sub(r"[^a-z0-9]+", " ", s).strip()
                return s

            for _, r in geo_df.iterrows():
                keys = []
                if str(r.get("feature_name", "")).strip():
                    keys.append(_norm_key(str(r.get("feature_name"))))
                if str(r.get("feature_id", "")).strip():
                    keys.append(_norm_key(str(r.get("feature_id"))))
                # Use both keys if present
                for k in keys:
                    if not k:
                        continue
                    if _is_number(r.get("area_m2", np.nan)):
                        geo_map_area[k] = float(r.get("area_m2"))
                    if _is_number(r.get("lat", np.nan)) and _is_number(r.get("lon", np.nan)):
                        geo_map_latlon[k] = (float(r.get("lat")), float(r.get("lon")))

            # Also allow substring fuzzy match later
            geo_keys = list(set(list(geo_map_area.keys()) + list(geo_map_latlon.keys())))
        else:
            geo_keys = []

        # Filter data to selected metric and build per-location series
        df_sig = df_long[df_long["metric_name"] == selected_metric].copy()
        if df_sig.empty:
            st.error("Selected metric has no rows. Choose a different metric in Analysis Settings.")
        else:
            # Determine timestamps presence
            has_time = df_sig["timestamp"].notna().any()
            # Build ambient time series if requested
            df_amb = pd.DataFrame()
            if baseline_method == "Use ambient metric from data (if available)" and ambient_metric and ambient_metric != "(none)":
                df_amb = df_long[df_long["metric_name"] == ambient_metric].copy()

            # Prepare ΔT estimation per record
            # For radiation term, we need an ambient value in °C. If not available, use ambient_constant_c.
            dt_records = []
            notes = []

            # Precompute per-building percentile baseline if needed
            pct_baseline = {}
            if not metric_is_deltaT and baseline_method == "Percentile baseline per building (10th percentile)":
                for loc, grp in df_sig.groupby("location_label"):
                    vals = pd.to_numeric(grp["metric_value"], errors="coerce").dropna()
                    if len(vals) > 5:
                        pct_baseline[loc] = float(np.percentile(vals, 10))
                    elif len(vals) > 0:
                        pct_baseline[loc] = float(np.min(vals))
                    else:
                        pct_baseline[loc] = ambient_constant_c

            # Build a join key for geo matching
            def _geo_lookup(location_label: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
                if not geo_keys:
                    return None, None, None
                lk = re.sub(r"[^a-z0-9]+", " ", _safe_lower(location_label)).strip()
                if not lk:
                    return None, None, None

                # Direct match
                area = geo_map_area.get(lk)
                latlon = geo_map_latlon.get(lk)

                # Substring fuzzy match (best effort)
                if area is None or latlon is None:
                    for gk in geo_keys[:1000]:
                        if gk and (gk in lk or lk in gk):
                            if area is None and gk in geo_map_area:
                                area = geo_map_area[gk]
                            if latlon is None and gk in geo_map_latlon:
                                latlon = geo_map_latlon[gk]
                            if area is not None and latlon is not None:
                                break

                lat = latlon[0] if latlon else None
                lon = latlon[1] if latlon else None
                return lat, lon, area

            # Build record-level deltaT and ambient
            if has_time:
                # Merge ambient by timestamp and location if possible
                if not df_amb.empty:
                    df_amb2 = df_amb[["timestamp", "location_label", "metric_value"]].rename(columns={"metric_value": "ambient_value"})
                    df_sig2 = df_sig.merge(df_amb2, on=["timestamp", "location_label"], how="left")
                else:
                    df_sig2 = df_sig.copy()
                    df_sig2["ambient_value"] = np.nan
            else:
                df_sig2 = df_sig.copy()
                df_sig2["ambient_value"] = np.nan

            # Apply area overrides and geo joins
            df_sig2["area_eff_m2"] = df_sig2.get("area_m2", np.nan)
            df_sig2["lat_eff"] = df_sig2.get("lat", np.nan)
            df_sig2["lon_eff"] = df_sig2.get("lon", np.nan)

            for loc, idxs in df_sig2.groupby("location_label").groups.items():
                # Area override first
                if loc in area_override_map:
                    df_sig2.loc[idxs, "area_eff_m2"] = area_override_map[loc]

                # If area still missing, try geo join
                if use_geo_join:
                    lat_j, lon_j, area_j = _geo_lookup(loc)
                    if area_j is not None and not np.isfinite(df_sig2.loc[idxs, "area_eff_m2"].astype(float)).any():
                        df_sig2.loc[idxs, "area_eff_m2"] = area_j
                    if lat_j is not None and lon_j is not None:
                        if not np.isfinite(df_sig2.loc[idxs, "lat_eff"].astype(float)).any():
                            df_sig2.loc[idxs, "lat_eff"] = lat_j
                        if not np.isfinite(df_sig2.loc[idxs, "lon_eff"].astype(float)).any():
                            df_sig2.loc[idxs, "lon_eff"] = lon_j

            # Fill missing area with default
            df_sig2["area_eff_m2"] = pd.to_numeric(df_sig2["area_eff_m2"], errors="coerce").fillna(default_area_m2)
            df_sig2["lat_eff"] = pd.to_numeric(df_sig2["lat_eff"], errors="coerce")
            df_sig2["lon_eff"] = pd.to_numeric(df_sig2["lon_eff"], errors="coerce")

            # Determine ambient for each record
            if baseline_method == "Use ambient metric from data (if available)" and not df_amb.empty:
                amb_used = pd.to_numeric(df_sig2["ambient_value"], errors="coerce")
                missing = amb_used.isna().sum()
                if missing > 0:
                    notes.append(f"Ambient metric missing for {missing:,} record(s); used constant ambient where needed.")
                df_sig2["ambient_c"] = amb_used.fillna(ambient_constant_c)
            elif baseline_method == "Percentile baseline per building (10th percentile)":
                df_sig2["ambient_c"] = df_sig2["location_label"].map(lambda x: pct_baseline.get(x, ambient_constant_c))
            else:
                df_sig2["ambient_c"] = ambient_constant_c

            # Compute ΔT
            vals = pd.to_numeric(df_sig2["metric_value"], errors="coerce")
            if metric_is_deltaT:
                df_sig2["delta_t_c"] = vals
            else:
                df_sig2["delta_t_c"] = vals - pd.to_numeric(df_sig2["ambient_c"], errors="coerce")

            # Clip negative deltas (no outward loss)
            df_sig2["delta_t_c_pos"] = df_sig2["delta_t_c"].clip(lower=0.0)

            # Data quality note
            if df_sig2["delta_t_c_pos"].dropna().empty:
                notes.append("ΔT is not positive for any records after baseline; heat loss will be ~0 under these settings.")

            # Aggregate per building
            results = []
            for loc, grp in df_sig2.groupby("location_label"):
                dts = pd.to_numeric(grp["delta_t_c_pos"], errors="coerce").dropna().values.astype(float)
                if len(dts) == 0:
                    dt_stat = np.nan
                else:
                    if agg_method == "Mean positive ΔT":
                        dt_stat = float(np.mean(dts))
                    elif agg_method == "Median positive ΔT":
                        dt_stat = float(np.median(dts))
                    elif agg_method == "90th percentile ΔT":
                        dt_stat = float(np.percentile(dts, 90))
                    else:
                        dt_stat = float(np.max(dts))

                # Ambient representative
                amb_vals = pd.to_numeric(grp["ambient_c"], errors="coerce").dropna().values.astype(float)
                t_amb_rep = float(np.median(amb_vals)) if len(amb_vals) else ambient_constant_c

                # Effective area
                area_rep = float(pd.to_numeric(grp["area_eff_m2"], errors="coerce").dropna().median()) if not grp.empty else default_area_m2

                # Central estimates
                q_wm2_central = _compute_heat_flux_wm2(
                    np.array([dt_stat], dtype=float),
                    np.array([t_amb_rep], dtype=float),
                    h=float(h_central),
                    emissivity=float(eps_central),
                )[0]
                power_w_central = q_wm2_central * area_rep
                annual_kwh_central = power_w_central * 8760.0 * float(seasonal_central) / 1000.0
                annual_cost_central = annual_kwh_central * float(tariff_gbp_per_kwh)
                annual_tco2_central = annual_kwh_central * float(carbon_kg_per_kwh) / 1000.0

                # Temporal variability
                dt_cv_time = None
                max_dt = None
                if len(dts) >= 5:
                    m = float(np.mean(dts))
                    s = float(np.std(dts))
                    dt_cv_time = (s / m) if m > 1e-9 else None
                    max_dt = float(np.max(dts))

                # Spatial variability is not reliably available without sub-location keys; try a proxy:
                # if multiple points have lat/lon for the same building, treat them as "spatial samples".
                spatial_cv = None
                if "lat_eff" in grp.columns and "lon_eff" in grp.columns:
                    # If there are multiple distinct points, treat their deltaT distribution as spatial mix
                    pts = grp[["lat_eff", "lon_eff", "delta_t_c_pos"]].dropna()
                    if len(pts) >= 10:
                        # Remove duplicates
                        pts2 = pts.drop_duplicates(subset=["lat_eff", "lon_eff"])
                        if len(pts2) >= 5:
                            xs = pts2["delta_t_c_pos"].values.astype(float)
                            m2 = float(np.mean(xs))
                            s2 = float(np.std(xs))
                            spatial_cv = (s2 / m2) if m2 > 1e-9 else None

                label = _classify_hotspot(
                    dt_median=float(np.median(dts)) if len(dts) else float(dt_stat) if np.isfinite(dt_stat) else np.nan,
                    dt_cv_time=dt_cv_time,
                    spatial_cv=spatial_cv,
                    max_dt=max_dt,
                )

                # Monte Carlo uncertainty bounds
                bounds = _monte_carlo_bounds(
                    delta_t_stat=float(dt_stat) if np.isfinite(dt_stat) else 0.0,
                    t_amb_c=float(t_amb_rep),
                    area_m2=float(area_rep),
                    tariff_gbp_per_kwh=float(tariff_gbp_per_kwh),
                    carbon_kg_per_kwh=float(carbon_kg_per_kwh),
                    h_central=float(h_central),
                    h_min=float(min(h_min, h_max)),
                    h_max=float(max(h_min, h_max)),
                    eps_central=float(eps_central),
                    eps_min=float(min(eps_min, eps_max)),
                    eps_max=float(max(eps_min, eps_max)),
                    seasonal_central=float(seasonal_central),
                    seasonal_min=float(min(seasonal_min, seasonal_max)),
                    seasonal_max=float(max(seasonal_min, seasonal_max)),
                    area_unc_frac=float(area_unc_frac),
                    amb_unc_c=float(amb_unc_c),
                    tariff_unc_frac=float(tariff_unc_frac),
                    carbon_unc_frac=float(carbon_unc_frac),
                    n=int(n_mc),
                    agg_method=str(agg_method),
                )

                # Representative coordinates
                lat_rep = pd.to_numeric(grp["lat_eff"], errors="coerce").dropna()
                lon_rep = pd.to_numeric(grp["lon_eff"], errors="coerce").dropna()
                lat_rep = float(lat_rep.median()) if len(lat_rep) else np.nan
                lon_rep = float(lon_rep.median()) if len(lon_rep) else np.nan

                results.append({
                    "location_label": loc,
                    "n_records": int(len(grp)),
                    "has_timestamps": bool(grp["timestamp"].notna().any()),
                    "metric_selected": selected_metric,
                    "metric_is_deltaT": bool(metric_is_deltaT),
                    "baseline_method": baseline_method,
                    "deltaT_stat_C": float(dt_stat) if np.isfinite(dt_stat) else 0.0,
                    "ambient_rep_C": float(t_amb_rep),
                    "area_eff_m2": float(area_rep),
                    "heat_flux_Wm2_central": float(q_wm2_central) if np.isfinite(q_wm2_central) else 0.0,
                    "power_W_central": float(power_w_central) if np.isfinite(power_w_central) else 0.0,
                    "annual_kWh_central": float(annual_kwh_central),
                    "annual_cost_GBP_central": float(annual_cost_central),
                    "annual_tCO2e_central": float(annual_tco2_central),
                    "classification": label,
                    "dt_cv_time": float(dt_cv_time) if dt_cv_time is not None else np.nan,
                    "spatial_cv_proxy": float(spatial_cv) if spatial_cv is not None else np.nan,
                    "max_deltaT_C": float(max_dt) if max_dt is not None else np.nan,
                    "lat": lat_rep,
                    "lon": lon_rep,
                    **bounds
                })

            res_df = pd.DataFrame(results)

            # Rank by avoidable annual cost (p50) if available, else central
            if "annual_cost_p50" in res_df.columns:
                res_df["rank_metric"] = res_df["annual_cost_p50"]
            else:
                res_df["rank_metric"] = res_df["annual_cost_GBP_central"]
            res_df = res_df.sort_values("rank_metric", ascending=False).reset_index(drop=True)
            res_df["rank"] = np.arange(1, len(res_df) + 1)

            st.session_state.analysis_results = res_df
            st.session_state.analysis_notes = notes

            # Display notes and ranked table
            if notes:
                with st.expander("Assumptions and warnings"):
                    for n in notes:
                        st.warning(n)

            st.markdown("#### Ranked buildings / zones")
            display_cols = [
                "rank",
                "location_label",
                "classification",
                "annual_kwh_p50",
                "annual_cost_p50",
                "annual_tco2_p50",
                "annual_cost_p10",
                "annual_cost_p90",
                "deltaT_stat_C",
                "area_eff_m2",
                "n_records",
            ]
            # Fall back if p50 columns missing (should exist)
            for c in display_cols:
                if c not in res_df.columns:
                    res_df[c] = np.nan

            st.dataframe(
                res_df[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")

            colL, colR = st.columns([0.9, 1.1], gap="large")

            with colL:
                st.markdown("#### Drill-down")
                selected_loc = st.selectbox(
                    "Select a building / zone:",
                    options=res_df["location_label"].tolist(),
                    index=0 if len(res_df) else 0,
                )
                row = res_df[res_df["location_label"] == selected_loc].iloc[0]

                st.write("**Key outputs (central / p50 / bounds):**")
                kpi = pd.DataFrame({
                    "Metric": ["Annual energy loss (kWh)", "Annual cost (£)", "Annual carbon (tCO₂e)"],
                    "Central": [row["annual_kWh_central"], row["annual_cost_GBP_central"], row["annual_tCO2e_central"]],
                    "P50": [row["annual_kwh_p50"], row["annual_cost_p50"], row["annual_tco2_p50"]],
                    "P10": [row["annual_kwh_p10"], row["annual_cost_p10"], row["annual_tco2_p10"]],
                    "P90": [row["annual_kwh_p90"], row["annual_cost_p90"], row["annual_tco2_p90"]],
                })
                st.dataframe(kpi, use_container_width=True, hide_index=True)

                st.write("**Interpretation:**")
                st.write(f"- Classification: **{row['classification']}**")
                st.write(f"- Aggregated ΔT used: **{row['deltaT_stat_C']:.2f} °C**")
                st.write(f"- Effective area: **{row['area_eff_m2']:.1f} m²**")
                st.write(f"- Convective h (central): **{h_central:.1f} W/m²·K** | Emissivity (central): **{eps_central:.2f}**")
                st.write(f"- Seasonal fraction (central): **{seasonal_central:.2f}**")

            with colR:
                st.markdown("#### Charts")

                # Distribution chart across buildings
                if len(res_df) > 1:
                    fig_hist = px.histogram(
                        res_df,
                        x="annual_cost_p50",
                        nbins=min(20, max(5, len(res_df) // 2)),
                        title="Distribution of annual avoidable cost (P50)",
                        labels={"annual_cost_p50": "Annual cost (£, P50)"},
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Time series plot if timestamps exist for selected location
                df_loc = df_sig2[df_sig2["location_label"] == selected_loc].copy()
                df_loc = df_loc.sort_values("timestamp") if "timestamp" in df_loc.columns else df_loc

                if df_loc["timestamp"].notna().any():
                    plot_df = df_loc[["timestamp", "delta_t_c_pos", "ambient_c"]].dropna(subset=["timestamp"])
                    # Downsample for performance
                    if len(plot_df) > 5000:
                        plot_df = plot_df.iloc[:: max(1, len(plot_df) // 5000)].copy()

                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=plot_df["timestamp"], y=plot_df["delta_t_c_pos"],
                        mode="lines", name="ΔT positive (°C)"
                    ))
                    fig_ts.add_trace(go.Scatter(
                        x=plot_df["timestamp"], y=plot_df["ambient_c"],
                        mode="lines", name="Ambient used (°C)"
                    ))
                    fig_ts.update_layout(
                        title="Time series (selected building)",
                        xaxis_title="Time",
                        yaxis_title="°C",
                        legend_title="Series",
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("No timestamps detected for the selected location. Time series chart is unavailable.")

            st.markdown("---")
            st.markdown("#### Map (blank background, offline-safe)")
            map_df = res_df.dropna(subset=["lat", "lon"]).copy()
            if map_df.empty:
                st.info("No usable lat/lon found. Load a GeoJSON with centroids, or include lat/lon metadata in JSON/CSV.")
            else:
                # Normalise for marker sizing
                mvals = pd.to_numeric(map_df["annual_cost_p50"], errors="coerce").fillna(0.0).values
                size = np.clip(np.sqrt(np.maximum(mvals, 0.0) + 1.0), 2.0, 30.0)
                map_df["size"] = size

                st.map(
                    map_df.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude", "size"]],
                    zoom=13,
                    use_container_width=True
                )


with tab_export:
    st.subheader("Export")
    res_df = st.session_state.analysis_results

    if res_df is None or res_df.empty:
        st.info("No results to export yet. Run analysis in the Results tab.")
    else:
        st.markdown("#### Download results")

        # CSV
        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="thermal_to_capex_results.csv",
            mime="text/csv",
        )

        # JSON
        payload = {
            "tool": "Thermal-to-CAPEX (EarthSavvy Add-on)",
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "notes": st.session_state.analysis_notes,
            "results": res_df.to_dict(orient="records"),
        }
        json_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
        st.download_button(
            "Download JSON",
            data=json_bytes,
            file_name="thermal_to_capex_results.json",
            mime="application/json",
        )

        # Offline HTML report (simple)
        st.markdown("#### Offline HTML report")
        top_n = st.slider("Top N rows in report", 5, min(100, len(res_df)), min(20, len(res_df)))

        # Create a compact summary
        top = res_df.head(top_n).copy()
        summary = {
            "n_buildings": int(len(res_df)),
            "total_cost_p50_gbp": float(np.nansum(pd.to_numeric(res_df["annual_cost_p50"], errors="coerce"))),
            "total_energy_p50_kwh": float(np.nansum(pd.to_numeric(res_df["annual_kwh_p50"], errors="coerce"))),
            "total_carbon_p50_tco2e": float(np.nansum(pd.to_numeric(res_df["annual_tco2_p50"], errors="coerce"))),
        }

        fig = px.bar(
            top,
            x="location_label",
            y="annual_cost_p50",
            title=f"Top {top_n} locations by annual cost (P50)",
            labels={"location_label": "Location", "annual_cost_p50": "Annual cost (£, P50)"},
        )
        fig.update_layout(xaxis_tickangle=-30)

        table_html = top[[
            "rank", "location_label", "classification",
            "annual_cost_p50", "annual_cost_p10", "annual_cost_p90",
            "annual_kwh_p50", "annual_tco2_p50",
            "deltaT_stat_C", "area_eff_m2"
        ]].to_html(index=False, escape=True)

        chart_html = pio.to_html(fig, include_plotlyjs="inline", full_html=False)

        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Thermal-to-CAPEX Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1 {{ margin-bottom: 6px; }}
.small {{ color: #444; font-size: 13px; }}
.kpis {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 12px 0 18px 0; }}
.kpi {{ border: 1px solid #ddd; padding: 10px 12px; border-radius: 8px; min-width: 220px; }}
.kpi .label {{ font-size: 12px; color: #555; }}
.kpi .value {{ font-size: 20px; font-weight: 600; margin-top: 2px; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }}
th {{ background: #f6f6f6; }}
.note {{ margin-top: 10px; font-size: 12px; color: #555; }}
</style>
</head>
<body>
<h1>Thermal-to-CAPEX (EarthSavvy Add-on)</h1>
<div class="small">Generated (UTC): {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</div>

<div class="kpis">
  <div class="kpi"><div class="label">Buildings analysed</div><div class="value">{summary["n_buildings"]}</div></div>
  <div class="kpi"><div class="label">Total annual cost (P50)</div><div class="value">£{summary["total_cost_p50_gbp"]:,.0f}</div></div>
  <div class="kpi"><div class="label">Total annual energy (P50)</div><div class="value">{summary["total_energy_p50_kwh"]:,.0f} kWh</div></div>
  <div class="kpi"><div class="label">Total annual carbon (P50)</div><div class="value">{summary["total_carbon_p50_tco2e"]:,.2f} tCO₂e</div></div>
</div>

<h2>Top locations (screening estimates)</h2>
{chart_html}

<h2>Ranked table (top {top_n})</h2>
{table_html}

<div class="note">
<b>Important:</b> This report is screening-grade and uses simplified heat-loss physics with uncertainty sampling. Results should be validated with site context, operational schedules, and (where possible) metered data.
</div>

</body>
</html>
"""
        st.download_button(
            "Download HTML report",
            data=html.encode("utf-8"),
            file_name="thermal_to_capex_report.html",
            mime="text/html",
        )

    st.markdown("---")
    st.markdown("#### requirements.txt (copy/paste)")
    st.code(DEFAULT_REQ_TXT.strip(), language="text")
