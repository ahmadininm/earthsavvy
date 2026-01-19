# code.py
"""
Thermal-to-CAPEX (EarthSavvy Add-on)
Single-file, offline Streamlit app.

How it works (screening-grade)
- You upload (or select) EarthSavvy exports (JSON), plus optional CSV/GeoJSON supporting files.
- The app auto-detects file types and parses them defensively into one canonical table:
  timestamp (optional), location_id/name, metric_name, metric_value, units (optional), geometry (optional), source_file
- It infers (where possible) surface temperature / ambient temperature / anomaly metrics, then estimates outward heat loss:
  q (W/m²) = h_conv * ΔT  +  εσ(Ts⁴ - Ta⁴)
- It converts W/m² to total W using area (from geometry where possible, otherwise a user default / override),
  then annualises to kWh using a chosen scaling method.
- It provides uncertainty bounds (P10/P50/P90) via Monte Carlo sampling of key parameters.
- It classifies each location using a simple, explainable ruleset:
  Likely Fabric Loss / Likely Process Hotspot / Mixed/Uncertain
- It ranks locations by avoidable annual energy, cost, and carbon impact, and exports CSV/JSON plus a simple HTML report.

Notes
- This is a screening tool, intended to justify where deeper investigation is worthwhile.
- It runs fully offline and continues with assumptions if key fields are missing.
"""

from __future__ import annotations

import os
import io
import re
import json
import math
import base64
import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional geometry support (recommended)
_HAS_SHAPELY = False
_HAS_PYPROJ = False
try:
    from shapely.geometry import shape as _shape
    from shapely.geometry.base import BaseGeometry as _BaseGeometry
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False

try:
    from pyproj import Geod as _Geod
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

SIGMA_SB = 5.670374419e-8  # W/m^2/K^4

APP_TITLE = "Thermal-to-CAPEX (EarthSavvy Add-on)"
ALLOWED_SCAN_EXTS = {".json", ".geojson", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".pdf", ".txt"}


# -----------------------------
# Utilities: file handling
# -----------------------------
@dataclass
class LoadedFile:
    name: str
    bytes_data: bytes
    source: str  # "upload" or "local"
    path: Optional[str] = None


def scan_working_directory() -> List[str]:
    """Return a sorted list of candidate files in the current working directory."""
    files = []
    try:
        for fn in os.listdir("."):
            if os.path.isfile(fn):
                ext = os.path.splitext(fn.lower())[1]
                if ext in ALLOWED_SCAN_EXTS:
                    files.append(fn)
    except Exception:
        return []
    return sorted(files, key=lambda x: x.lower())


def sniff_file_type(name: str, raw: bytes) -> str:
    """Lightweight file type detection."""
    ext = os.path.splitext(name.lower())[1]
    if ext in {".geojson"}:
        return "geojson"
    if ext in {".json"}:
        # Could still be GeoJSON
        try:
            obj = json.loads(raw.decode("utf-8", errors="ignore"))
            if isinstance(obj, dict) and "features" in obj and obj.get("type", "").lower() == "featurecollection":
                return "geojson"
        except Exception:
            pass
        return "json"
    if ext in {".csv"}:
        return "csv"
    if ext in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if ext in {".pdf"}:
        return "pdf"
    return "other"


def safe_json_loads(raw: bytes) -> Optional[Any]:
    try:
        return json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        return None


def safe_read_csv(raw: bytes) -> Optional[pd.DataFrame]:
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    # Last attempt with pandas default
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None


def try_parse_datetime(x: Any) -> Optional[pd.Timestamp]:
    if x is None:
        return None
    if isinstance(x, (pd.Timestamp, _dt.datetime)):
        return pd.Timestamp(x)
    if isinstance(x, (int, float)) and not np.isnan(x):
        # Heuristic: unix seconds or ms
        if x > 1e12:
            # ms
            try:
                return pd.to_datetime(int(x), unit="ms", utc=True).tz_convert(None)
            except Exception:
                return None
        if x > 1e9:
            # seconds
            try:
                return pd.to_datetime(int(x), unit="s", utc=True).tz_convert(None)
            except Exception:
                return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # Common formats: "YYYY-MM-DD HH:MM", ISO, etc.
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return None
    return None


def is_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        if isinstance(x, bool):
            return False
        v = float(x)
        return np.isfinite(v)
    except Exception:
        return False


def flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, kk, sep=sep))
        else:
            out[kk] = v
    return out


# -----------------------------
# Canonical records + parsing
# -----------------------------
CANON_COLS = ["timestamp", "location_id", "location_name", "metric_name", "metric_value", "units", "geometry", "source_file"]


def canonical_records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=CANON_COLS)
    df = pd.DataFrame(records)
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[CANON_COLS + [c for c in df.columns if c not in CANON_COLS]]
    # Normalise types
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    # timestamp can be NaT
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def parse_geojson_featurecollection(obj: Dict[str, Any], source_file: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Parse GeoJSON FeatureCollection.
    Returns (canonical_metric_records, geometry_df)
    """
    records: List[Dict[str, Any]] = []
    geoms: List[Dict[str, Any]] = []

    features = obj.get("features", [])
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties", {}) if isinstance(feat.get("properties", {}), dict) else {}
        geom = feat.get("geometry")
        loc_id = props.get("id") or props.get("location_id") or props.get("name") or f"feature_{i}"
        loc_name = props.get("name") or props.get("location_name") or str(loc_id)

        # Geometry details
        area_m2 = None
        centroid_lat = None
        centroid_lon = None
        if geom is not None:
            area_m2, centroid_lat, centroid_lon = estimate_area_and_centroid_from_geojson(geom)

        geoms.append(
            {
                "location_id": str(loc_id),
                "location_name": str(loc_name),
                "geometry": geom,
                "area_m2": area_m2,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "source_file": source_file,
            }
        )

        # Numeric properties as metrics
        flat_props = flatten_dict(props)
        for k, v in flat_props.items():
            if is_number(v):
                records.append(
                    {
                        "timestamp": None,
                        "location_id": str(loc_id),
                        "location_name": str(loc_name),
                        "metric_name": str(k),
                        "metric_value": float(v),
                        "units": None,
                        "geometry": geom,
                        "source_file": source_file,
                    }
                )

    geom_df = pd.DataFrame(geoms) if geoms else pd.DataFrame(columns=["location_id", "location_name", "geometry", "area_m2", "centroid_lat", "centroid_lon", "source_file"])
    return records, geom_df


def parse_json_records(obj: Any, source_file: str) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Robust EarthSavvy-ish JSON parser.
    Returns (canonical_metric_records, geometry_df)
    """
    records: List[Dict[str, Any]] = []
    geom_df = pd.DataFrame(columns=["location_id", "location_name", "geometry", "area_m2", "centroid_lat", "centroid_lon", "source_file"])

    if isinstance(obj, dict) and obj.get("type", "").lower() == "featurecollection" and "features" in obj:
        recs, gdf = parse_geojson_featurecollection(obj, source_file)
        return recs, gdf

    # Pattern A: dict keyed by location/site, each entry with {"name":..., "data":[...]}
    if isinstance(obj, dict):
        # If it looks like an EarthSavvy export containing per-site dicts
        site_like_keys = [k for k, v in obj.items() if isinstance(v, dict) and ("data" in v or "timeseries" in v)]
        if site_like_keys:
            for k in site_like_keys:
                site = obj.get(k, {})
                loc_id = site.get("id") or site.get("location_id") or k
                loc_name = site.get("name") or site.get("location_name") or str(loc_id)

                # Any geometry attached at this level
                site_geom = site.get("geometry") or site.get("geojson") or None
                if site_geom is not None:
                    area_m2, clat, clon = estimate_area_and_centroid_from_geojson(site_geom)
                    geom_df = pd.concat(
                        [
                            geom_df,
                            pd.DataFrame(
                                [
                                    {
                                        "location_id": str(loc_id),
                                        "location_name": str(loc_name),
                                        "geometry": site_geom,
                                        "area_m2": area_m2,
                                        "centroid_lat": clat,
                                        "centroid_lon": clon,
                                        "source_file": source_file,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

                # data arrays
                data = site.get("data") or site.get("timeseries")
                if isinstance(data, list) and data:
                    # list of lists or list of dicts
                    if all(isinstance(r, list) for r in data):
                        # Try infer first column timestamp
                        for row in data:
                            if not row:
                                continue
                            t0 = try_parse_datetime(row[0])
                            if t0 is not None and pd.notna(t0):
                                # row[1:] are metrics
                                for j in range(1, len(row)):
                                    val = row[j]
                                    if is_number(val):
                                        records.append(
                                            {
                                                "timestamp": t0,
                                                "location_id": str(loc_id),
                                                "location_name": str(loc_name),
                                                "metric_name": f"metric_{j}",
                                                "metric_value": float(val),
                                                "units": None,
                                                "geometry": site_geom,
                                                "source_file": source_file,
                                            }
                                        )
                            else:
                                # treat as static metrics vector for this location
                                for j, val in enumerate(row):
                                    if is_number(val):
                                        records.append(
                                            {
                                                "timestamp": None,
                                                "location_id": str(loc_id),
                                                "location_name": str(loc_name),
                                                "metric_name": f"metric_{j}",
                                                "metric_value": float(val),
                                                "units": None,
                                                "geometry": site_geom,
                                                "source_file": source_file,
                                            }
                                        )
                    elif all(isinstance(r, dict) for r in data):
                        for r in data:
                            flat = flatten_dict(r)
                            ts = None
                            for cand in ["timestamp", "time", "datetime", "date"]:
                                if cand in flat:
                                    ts = try_parse_datetime(flat[cand])
                                    break
                            loc_id2 = flat.get("location_id") or flat.get("id") or loc_id
                            loc_name2 = flat.get("location_name") or flat.get("name") or loc_name
                            units = flat.get("units") or None
                            for kk, vv in flat.items():
                                if kk.lower() in {"timestamp", "time", "datetime", "date", "location_id", "id", "location_name", "name", "units"}:
                                    continue
                                if is_number(vv):
                                    records.append(
                                        {
                                            "timestamp": ts,
                                            "location_id": str(loc_id2),
                                            "location_name": str(loc_name2),
                                            "metric_name": str(kk),
                                            "metric_value": float(vv),
                                            "units": units,
                                            "geometry": site_geom,
                                            "source_file": source_file,
                                        }
                                    )

                # other numeric fields at site level
                flat_site = flatten_dict(site)
                for kk, vv in flat_site.items():
                    if kk in {"data", "timeseries"}:
                        continue
                    if isinstance(vv, (dict, list)):
                        continue
                    if is_number(vv):
                        records.append(
                            {
                                "timestamp": None,
                                "location_id": str(loc_id),
                                "location_name": str(loc_name),
                                "metric_name": str(kk),
                                "metric_value": float(vv),
                                "units": None,
                                "geometry": site_geom,
                                "source_file": source_file,
                            }
                        )
            return records, geom_df

    # Pattern B: list of dict records
    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        for i, rec in enumerate(obj):
            flat = flatten_dict(rec)
            ts = None
            for cand in ["timestamp", "time", "datetime", "date"]:
                if cand in flat:
                    ts = try_parse_datetime(flat[cand])
                    break
            loc_id = flat.get("location_id") or flat.get("id") or flat.get("site_id") or flat.get("building_id") or f"row_{i}"
            loc_name = flat.get("location_name") or flat.get("name") or flat.get("site") or flat.get("building") or str(loc_id)
            units = flat.get("units") or None
            geom = rec.get("geometry") if isinstance(rec.get("geometry"), dict) else None

            if geom is not None:
                area_m2, clat, clon = estimate_area_and_centroid_from_geojson(geom)
                geom_df = pd.concat(
                    [
                        geom_df,
                        pd.DataFrame(
                            [
                                {
                                    "location_id": str(loc_id),
                                    "location_name": str(loc_name),
                                    "geometry": geom,
                                    "area_m2": area_m2,
                                    "centroid_lat": clat,
                                    "centroid_lon": clon,
                                    "source_file": source_file,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            # numeric fields as metrics
            for kk, vv in flat.items():
                if kk.lower() in {"timestamp", "time", "datetime", "date", "location_id", "id", "site_id", "building_id", "location_name", "name", "site", "building", "units"}:
                    continue
                if is_number(vv):
                    records.append(
                        {
                            "timestamp": ts,
                            "location_id": str(loc_id),
                            "location_name": str(loc_name),
                            "metric_name": str(kk),
                            "metric_value": float(vv),
                            "units": units,
                            "geometry": geom,
                            "source_file": source_file,
                        }
                    )
        return records, geom_df

    # Pattern C: arbitrary dict with some time series arrays nested
    if isinstance(obj, dict):
        # Try to find nested arrays of records and recurse
        def _walk(o: Any, path: str = ""):
            if isinstance(o, dict):
                # If it looks like a metric bundle: {"name":..., "units":..., "data":[...]}
                if "data" in o and isinstance(o["data"], list):
                    # Attempt parse as timeseries-like
                    name = o.get("name") or o.get("metric") or path or "metric"
                    units = o.get("units") or None
                    data = o["data"]
                    if data and all(isinstance(r, list) for r in data):
                        for r in data:
                            if not r:
                                continue
                            ts = try_parse_datetime(r[0])
                            if ts is not None and pd.notna(ts):
                                if len(r) == 2 and is_number(r[1]):
                                    records.append(
                                        {
                                            "timestamp": ts,
                                            "location_id": "unknown",
                                            "location_name": "unknown",
                                            "metric_name": str(name),
                                            "metric_value": float(r[1]),
                                            "units": units,
                                            "geometry": None,
                                            "source_file": source_file,
                                        }
                                    )
                                else:
                                    # multiple columns
                                    for j in range(1, len(r)):
                                        if is_number(r[j]):
                                            records.append(
                                                {
                                                    "timestamp": ts,
                                                    "location_id": "unknown",
                                                    "location_name": "unknown",
                                                    "metric_name": f"{name}.metric_{j}",
                                                    "metric_value": float(r[j]),
                                                    "units": units,
                                                    "geometry": None,
                                                    "source_file": source_file,
                                                }
                                            )
                for k, v in o.items():
                    _walk(v, f"{path}.{k}" if path else str(k))
            elif isinstance(o, list):
                for j, it in enumerate(o):
                    _walk(it, f"{path}[{j}]")

        _walk(obj)
        # If we found anything, return it
        if records:
            return records, geom_df

        # Otherwise, flatten numeric fields as static metrics
        flat = flatten_dict(obj)
        for kk, vv in flat.items():
            if is_number(vv):
                records.append(
                    {
                        "timestamp": None,
                        "location_id": "unknown",
                        "location_name": "unknown",
                        "metric_name": str(kk),
                        "metric_value": float(vv),
                        "units": None,
                        "geometry": None,
                        "source_file": source_file,
                    }
                )
        return records, geom_df

    return records, geom_df


def parse_csv_to_canonical(df: pd.DataFrame, source_file: str) -> List[Dict[str, Any]]:
    """
    Convert a generic CSV table into canonical metric records.
    Heuristics:
    - Find timestamp column: timestamp/time/datetime/date
    - Find location column: location_id/location/name/site/building
    - All other numeric columns become metrics
    """
    if df is None or df.empty:
        return []

    cols_lower = {c.lower(): c for c in df.columns}
    ts_col = None
    for cand in ["timestamp", "time", "datetime", "date"]:
        if cand in cols_lower:
            ts_col = cols_lower[cand]
            break

    loc_id_col = None
    for cand in ["location_id", "site_id", "building_id", "id"]:
        if cand in cols_lower:
            loc_id_col = cols_lower[cand]
            break

    loc_name_col = None
    for cand in ["location_name", "name", "site", "building"]:
        if cand in cols_lower:
            loc_name_col = cols_lower[cand]
            break

    # Identify numeric columns
    numeric_cols = []
    for c in df.columns:
        if c in {ts_col, loc_id_col, loc_name_col}:
            continue
        # Try convert to numeric quickly
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            numeric_cols.append(c)

    records: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        ts = try_parse_datetime(row[ts_col]) if ts_col else None
        loc_id = str(row[loc_id_col]) if loc_id_col else "unknown"
        loc_name = str(row[loc_name_col]) if loc_name_col else loc_id
        for c in numeric_cols:
            val = row.get(c)
            if is_number(val):
                records.append(
                    {
                        "timestamp": ts,
                        "location_id": loc_id,
                        "location_name": loc_name,
                        "metric_name": str(c),
                        "metric_value": float(val),
                        "units": None,
                        "geometry": None,
                        "source_file": source_file,
                    }
                )
    return records


# -----------------------------
# Geometry area + centroid
# -----------------------------
def estimate_area_and_centroid_from_geojson(geom: Any) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Estimate area (m²) and centroid (lat, lon) for GeoJSON geometry (Polygon/MultiPolygon).
    - If shapely + pyproj available, compute geodesic area using pyproj.Geod
    - Otherwise: area None; centroid heuristic for point-ish geometries
    """
    if geom is None:
        return None, None, None

    # Try centroid quickly even without shapely
    centroid_lat = None
    centroid_lon = None

    if isinstance(geom, dict):
        gtype = (geom.get("type") or "").lower()
        coords = geom.get("coordinates")

        if gtype == "point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            centroid_lon, centroid_lat = float(coords[0]), float(coords[1])
            return None, centroid_lat, centroid_lon

    # Full path if available
    if _HAS_SHAPELY and _HAS_PYPROJ and isinstance(geom, dict):
        try:
            s = _shape(geom)
            if not isinstance(s, _BaseGeometry) or s.is_empty:
                return None, None, None
            centroid = s.centroid
            centroid_lon, centroid_lat = float(centroid.x), float(centroid.y)

            # Geodesic area
            geod = _Geod(ellps="WGS84")
            area = geodesic_area_m2(geod, geom)
            if area is not None:
                area = abs(area)
            return area, centroid_lat, centroid_lon
        except Exception:
            pass

    # Fallback centroid for polygons (rough)
    if isinstance(geom, dict) and isinstance(geom.get("coordinates"), list):
        try:
            coords = geom["coordinates"]
            pts = []
            # GeoJSON polygon: [ [ [lon,lat],... ] ] ; multipolygon: [ [ [ [lon,lat],... ] ] ]
            def _collect(c):
                if isinstance(c, (list, tuple)) and len(c) == 2 and all(is_number(x) for x in c):
                    pts.append((float(c[0]), float(c[1])))
                elif isinstance(c, (list, tuple)):
                    for it in c:
                        _collect(it)

            _collect(coords)
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                centroid_lon = float(np.mean(xs))
                centroid_lat = float(np.mean(ys))
        except Exception:
            pass

    return None, centroid_lat, centroid_lon


def geodesic_area_m2(geod: "_Geod", geom: Dict[str, Any]) -> Optional[float]:
    """Compute geodesic area for Polygon/MultiPolygon GeoJSON."""
    try:
        gtype = (geom.get("type") or "").lower()
        coords = geom.get("coordinates")
        if gtype == "polygon":
            # coords: [ ring1, ring2 (holes), ... ], ring: [ [lon,lat], ... ]
            return polygon_area_geod(geod, coords)
        if gtype == "multipolygon":
            area = 0.0
            for poly in coords:
                area += (polygon_area_geod(geod, poly) or 0.0)
            return area
    except Exception:
        return None
    return None


def polygon_area_geod(geod: "_Geod", poly_coords: Any) -> Optional[float]:
    """poly_coords: list of rings; returns signed area (m²)."""
    try:
        if not isinstance(poly_coords, list) or not poly_coords:
            return None
        total_area = 0.0
        for ring in poly_coords:
            if not isinstance(ring, list) or len(ring) < 3:
                continue
            lons = [p[0] for p in ring if isinstance(p, (list, tuple)) and len(p) >= 2]
            lats = [p[1] for p in ring if isinstance(p, (list, tuple)) and len(p) >= 2]
            if len(lons) < 3:
                continue
            a, _ = geod.polygon_area_perimeter(lons, lats)
            total_area += a
        return total_area
    except Exception:
        return None


# -----------------------------
# Metric inference helpers
# -----------------------------
def normalise_metric_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")


def infer_metric_columns(cdf: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Infer likely metric_name candidates for:
    - surface temperature
    - ambient temperature
    - anomaly / deltaT
    - footprint area
    """
    names = sorted(set(cdf["metric_name"].dropna().astype(str).tolist()))
    norm_map = {n: normalise_metric_name(n) for n in names}

    surface = []
    ambient = []
    anomaly = []
    area = []
    hotspot = []

    for orig, nn in norm_map.items():
        # Surface temperature
        if ("surface" in nn or "skin" in nn or "roof" in nn or "external" in nn) and ("temp" in nn or nn.endswith("_t")):
            surface.append(orig)
        if ("temp" in nn) and ("surface" in nn or "ts" == nn or nn.endswith("_ts")):
            surface.append(orig)

        # Ambient temperature
        if ("ambient" in nn or "air" in nn or "outside" in nn or "background" in nn) and ("temp" in nn or nn.endswith("_t")):
            ambient.append(orig)

        # Anomaly
        if ("anomaly" in nn) or ("delta" in nn and "temp" in nn) or (nn in {"dt", "d_t", "deltat"}):
            anomaly.append(orig)
        if ("temp" in nn and ("diff" in nn or "difference" in nn)):
            anomaly.append(orig)

        # Area
        if ("area" in nn) or ("footprint" in nn) or (nn.endswith("_m2")):
            area.append(orig)

        # Hotspot indicators
        if ("hotspot" in nn) or ("max" in nn and "temp" in nn) or ("peak" in nn and "temp" in nn):
            hotspot.append(orig)

    # De-duplicate while preserving order
    def _uniq(lst):
        out = []
        seen = set()
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "surface_temp": _uniq(surface),
        "ambient_temp": _uniq(ambient),
        "anomaly": _uniq(anomaly),
        "area": _uniq(area),
        "hotspot": _uniq(hotspot),
    }


def pick_best_metric(candidates: List[str], fallback: Optional[str] = None) -> Optional[str]:
    if candidates:
        return candidates[0]
    return fallback


# -----------------------------
# Heat-loss model + uncertainty
# -----------------------------
def heat_flux_w_m2(delta_t_c: np.ndarray, ambient_c: np.ndarray, h_conv: float, emissivity: float) -> np.ndarray:
    """
    Outward heat flux from a warmer surface to ambient:
      q = h_conv * ΔT + εσ(Ts^4 - Ta^4)
    Temperatures are in Celsius; converted to Kelvin for radiation.
    Negative ΔT is clipped to zero (no outward loss).
    """
    dt = np.maximum(delta_t_c, 0.0)
    ta_k = ambient_c + 273.15
    ts_k = ta_k + dt
    q_conv = h_conv * dt
    q_rad = emissivity * SIGMA_SB * (np.power(ts_k, 4) - np.power(ta_k, 4))
    q = q_conv + q_rad
    q = np.maximum(q, 0.0)
    return q


def linearised_h_rad_w_m2k(ambient_c: np.ndarray, delta_t_c: np.ndarray, emissivity: float) -> np.ndarray:
    """
    Linearised radiative heat transfer coefficient around a mid-temperature:
      h_rad ≈ 4 ε σ Tm^3
    with Tm in Kelvin.
    """
    dt = np.maximum(delta_t_c, 0.0)
    ta_k = ambient_c + 273.15
    tm_k = ta_k + 0.5 * dt
    h_rad = 4.0 * emissivity * SIGMA_SB * np.power(tm_k, 3)
    h_rad = np.maximum(h_rad, 0.0)
    return h_rad


def annualise_energy_kwh(
    q_w_m2: np.ndarray,
    area_m2: float,
    method: str,
    hours_per_year: float,
    season_fraction: float,
    annual_hdd_kdays: float,
    ambient_c: np.ndarray,
    delta_t_c: np.ndarray,
    h_conv: float,
    emissivity: float,
) -> np.ndarray:
    """
    Convert heat flux to annual energy kWh (vectorised over samples).
    """
    if area_m2 is None or not np.isfinite(area_m2) or area_m2 <= 0:
        area_m2 = 1.0

    power_w = q_w_m2 * area_m2

    method = (method or "").strip().lower()
    if method == "fixed runtime hours":
        e_kwh = power_w * hours_per_year / 1000.0
        return np.maximum(e_kwh, 0.0)

    if method == "seasonal fraction":
        e_kwh = power_w * (8760.0 * season_fraction) / 1000.0
        return np.maximum(e_kwh, 0.0)

    if method == "degree-day (hdd)":
        # Estimate UA using linearised combined heat transfer coefficient
        dt = np.maximum(delta_t_c, 0.0)
        # Avoid divide-by-zero if no temperature rise
        dt_safe = np.where(dt > 0.1, dt, 0.1)
        # Effective h_total ~ h_conv + h_rad
        h_rad = linearised_h_rad_w_m2k(ambient_c, dt, emissivity)
        h_total = h_conv + h_rad
        ua_w_per_k = h_total * area_m2
        # Annual HDD in K*days -> multiply by 24 h/day
        e_kwh = ua_w_per_k * (annual_hdd_kdays * 24.0) / 1000.0
        # For very low ΔT cases, UA may still be large; this is screening-only.
        return np.maximum(e_kwh, 0.0)

    # Default: "seasonal fraction"
    e_kwh = power_w * (8760.0 * season_fraction) / 1000.0
    return np.maximum(e_kwh, 0.0)


def compute_uncertainty_bounds(
    dt_samples: np.ndarray,
    ambient_samples: np.ndarray,
    area_m2: float,
    tariff_gbp_per_kwh: float,
    carbon_kg_per_kwh: float,
    method: str,
    hours_per_year: float,
    season_fraction: float,
    annual_hdd_kdays: float,
    n_mc: int,
    h_conv_central: float,
    h_conv_rel_range: float,
    emissivity_central: float,
    emissivity_abs_range: float,
    area_rel_range: float,
    ambient_abs_range_c: float,
    dt_abs_range_c: float,
    scaling_rel_range: float,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Monte Carlo sampling for uncertainty bounds.
    Returns percentiles for energy/cost/carbon.
    """
    n_mc = int(max(200, min(5000, n_mc)))

    # Sample ΔT by bootstrap (if multiple points) else by normal noise
    dt_samples = np.asarray(dt_samples, dtype=float)
    dt_samples = dt_samples[np.isfinite(dt_samples)]
    if dt_samples.size == 0:
        dt0 = np.array([0.0])
    else:
        dt0 = dt_samples

    if dt0.size >= 5:
        idx = rng.integers(0, dt0.size, size=n_mc)
        dt_mc = dt0[idx]
        dt_mc = dt_mc + rng.normal(0.0, dt_abs_range_c, size=n_mc)
    else:
        central = float(np.nanmean(dt0)) if dt0.size else 0.0
        dt_mc = rng.normal(central, max(0.5, dt_abs_range_c), size=n_mc)

    dt_mc = np.maximum(dt_mc, 0.0)

    # Ambient
    ambient_samples = np.asarray(ambient_samples, dtype=float)
    ambient_samples = ambient_samples[np.isfinite(ambient_samples)]
    if ambient_samples.size >= 5:
        idxa = rng.integers(0, ambient_samples.size, size=n_mc)
        amb_mc = ambient_samples[idxa] + rng.normal(0.0, ambient_abs_range_c, size=n_mc)
    elif ambient_samples.size >= 1:
        amb_mc = rng.normal(float(np.nanmean(ambient_samples)), max(0.5, ambient_abs_range_c), size=n_mc)
    else:
        amb_mc = rng.normal(10.0, max(1.0, ambient_abs_range_c), size=n_mc)

    # Parameters
    h_low = max(0.5, h_conv_central * (1.0 - h_conv_rel_range))
    h_high = h_conv_central * (1.0 + h_conv_rel_range)
    h_mc = rng.uniform(h_low, h_high, size=n_mc)

    e_low = max(0.5, emissivity_central - emissivity_abs_range)
    e_high = min(1.0, emissivity_central + emissivity_abs_range)
    eps_mc = rng.uniform(e_low, e_high, size=n_mc)

    # Area (lognormal-ish)
    a0 = float(area_m2) if area_m2 and np.isfinite(area_m2) and area_m2 > 0 else 1.0
    a_low = max(0.1, a0 * (1.0 - area_rel_range))
    a_high = a0 * (1.0 + area_rel_range)
    area_mc = rng.uniform(a_low, a_high, size=n_mc)

    # Base heat flux
    q_mc = heat_flux_w_m2(dt_mc, amb_mc, h_mc, eps_mc)

    # Annualisation
    e_kwh_mc = annualise_energy_kwh(
        q_w_m2=q_mc,
        area_m2=area_mc,
        method=method,
        hours_per_year=hours_per_year,
        season_fraction=season_fraction,
        annual_hdd_kdays=annual_hdd_kdays,
        ambient_c=amb_mc,
        delta_t_c=dt_mc,
        h_conv=float(h_conv_central),
        emissivity=float(emissivity_central),
    )

    # Scaling uncertainty (multiplicative)
    scale_low = max(0.1, 1.0 - scaling_rel_range)
    scale_high = 1.0 + scaling_rel_range
    scale_mc = rng.uniform(scale_low, scale_high, size=n_mc)
    e_kwh_mc = e_kwh_mc * scale_mc

    cost_mc = e_kwh_mc * float(tariff_gbp_per_kwh)
    carbon_t_mc = e_kwh_mc * float(carbon_kg_per_kwh) / 1000.0

    def pct(x):
        return {
            "p10": float(np.nanpercentile(x, 10)),
            "p50": float(np.nanpercentile(x, 50)),
            "p90": float(np.nanpercentile(x, 90)),
        }

    return {
        "energy_kwh": pct(e_kwh_mc),
        "cost_gbp": pct(cost_mc),
        "carbon_tco2e": pct(carbon_t_mc),
        "n_mc": n_mc,
    }


# -----------------------------
# Classification rules
# -----------------------------
def classify_location(dt_series: np.ndarray, hotspot_ratio: Optional[float], data_richness: int) -> Tuple[str, str]:
    """
    Explainable ruleset based on ΔT variability and peakiness.
    Returns (label, reason)
    """
    dt_series = np.asarray(dt_series, dtype=float)
    dt_series = dt_series[np.isfinite(dt_series)]
    if dt_series.size < 3:
        return "Uncertain", "Insufficient time series coverage for classification."

    mean_dt = float(np.mean(dt_series))
    std_dt = float(np.std(dt_series))
    if mean_dt <= 0.2:
        return "Uncertain", "Temperature anomaly is near zero on average."

    cv = std_dt / max(mean_dt, 1e-6)
    p95 = float(np.percentile(dt_series, 95))
    p50 = float(np.percentile(dt_series, 50))
    peakiness = p95 / max(p50, 1e-6)

    # Hotspot ratio (if available) biases towards process hotspots
    hr = float(hotspot_ratio) if hotspot_ratio is not None and np.isfinite(hotspot_ratio) else None

    # Rule interpretation:
    # - Fabric losses tend to be persistent and broad: low CV, lower peakiness.
    # - Process hotspots tend to be intermittent or localised: higher peakiness, higher variability.
    # - Mixed/Uncertain in between.
    if data_richness < 2:
        # Be conservative when data is sparse
        if cv < 0.25 and peakiness < 1.35:
            return "Likely Fabric Loss", "Persistent anomaly with low variability (sparse data)."
        if cv > 0.60 or peakiness > 2.0:
            return "Likely Process Hotspot", "Highly variable anomaly with pronounced peaks (sparse data)."
        return "Mixed/Uncertain", "Intermediate behaviour (sparse data)."

    # Richer data
    if cv < 0.25 and peakiness < 1.35 and (hr is None or hr < 1.6):
        return "Likely Fabric Loss", "Persistent anomaly with low variability and limited peak behaviour."
    if (cv > 0.55 or peakiness > 1.9) and (hr is None or hr >= 1.3):
        return "Likely Process Hotspot", "High variability and/or peak behaviour suggests localised or intermittent heat sources."
    return "Mixed/Uncertain", "Anomaly shows both persistent and peaky characteristics."


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.caption("Offline screening tool for turning EarthSavvy thermal evidence into CAPEX-ready heat loss estimates with uncertainty bounds.")

# Session state initialisation
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files: List[LoadedFile] = []
if "canonical_df" not in st.session_state:
    st.session_state.canonical_df = pd.DataFrame(columns=CANON_COLS)
if "geom_df" not in st.session_state:
    st.session_state.geom_df = pd.DataFrame(columns=["location_id", "location_name", "geometry", "area_m2", "centroid_lat", "centroid_lon", "source_file"])
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "assumptions" not in st.session_state:
    st.session_state.assumptions = {}
if "inferred_metrics" not in st.session_state:
    st.session_state.inferred_metrics = {}


tab_upload, tab_settings, tab_results, tab_export = st.tabs(["Upload & Detect", "Analysis Settings", "Results", "Export"])


# -----------------------------
# Tab 1: Upload & Detect
# -----------------------------
with tab_upload:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Upload files")
        up = st.file_uploader(
            "Upload one or more files (JSON/CSV/GeoJSON; optional images/PDFs for context).",
            type=None,
            accept_multiple_files=True,
        )

        st.divider()
        st.subheader("Or select from this folder")
        local_files = scan_working_directory()
        selected_local = st.multiselect(
            "Files detected in the working directory (same folder as code.py).",
            options=local_files,
            default=[],
        )

        st.divider()
        parse_now = st.button("Parse selected files", type="primary")

    with right:
        st.subheader("Detected files and quick preview")
        if up:
            st.write(f"Uploads selected: **{len(up)}**")
        if selected_local:
            st.write(f"Local files selected: **{len(selected_local)}**")

        # Show previews for context files (image/pdf)
        preview_candidates: List[LoadedFile] = []

        if up:
            for f in up[:3]:
                preview_candidates.append(LoadedFile(name=f.name, bytes_data=f.getvalue(), source="upload"))
        for fn in selected_local[:3]:
            try:
                with open(fn, "rb") as fh:
                    preview_candidates.append(LoadedFile(name=fn, bytes_data=fh.read(), source="local", path=fn))
            except Exception:
                continue

        for lf in preview_candidates:
            ftype = sniff_file_type(lf.name, lf.bytes_data)
            st.markdown(f"**{lf.name}**  \nType: `{ftype}`")
            if ftype == "image":
                st.image(lf.bytes_data, use_container_width=True)
            elif ftype == "pdf":
                st.info("PDF detected. (Preview is not rendered here, but it is kept for reference.)")
            else:
                # Text preview
                try:
                    txt = lf.bytes_data.decode("utf-8", errors="ignore")
                    st.code(txt[:600] + ("..." if len(txt) > 600 else ""))
                except Exception:
                    st.code(f"Binary file ({len(lf.bytes_data)} bytes).")
            st.divider()

    if parse_now:
        loaded: List[LoadedFile] = []

        # Add uploads
        if up:
            for f in up:
                loaded.append(LoadedFile(name=f.name, bytes_data=f.getvalue(), source="upload"))

        # Add local selections
        for fn in selected_local:
            try:
                with open(fn, "rb") as fh:
                    loaded.append(LoadedFile(name=fn, bytes_data=fh.read(), source="local", path=fn))
            except Exception as e:
                st.warning(f"Could not read local file: {fn} ({e})")

        if not loaded:
            st.warning("No files selected. Please upload and/or select local files.")
        else:
            st.session_state.loaded_files = loaded

            all_records: List[Dict[str, Any]] = []
            all_geoms: List[pd.DataFrame] = []

            issues = []

            for lf in loaded:
                ftype = sniff_file_type(lf.name, lf.bytes_data)
                try:
                    if ftype in {"json", "geojson"}:
                        obj = safe_json_loads(lf.bytes_data)
                        if obj is None:
                            issues.append(f"{lf.name}: could not parse JSON.")
                            continue
                        recs, gdf = parse_json_records(obj, source_file=lf.name)
                        all_records.extend(recs)
                        if gdf is not None and not gdf.empty:
                            all_geoms.append(gdf)

                    elif ftype == "csv":
                        df = safe_read_csv(lf.bytes_data)
                        if df is None:
                            issues.append(f"{lf.name}: could not parse CSV.")
                            continue
                        recs = parse_csv_to_canonical(df, source_file=lf.name)
                        all_records.extend(recs)

                    else:
                        # Context only (image/pdf/other). Keep but do not parse into canonical metrics.
                        continue
                except Exception as e:
                    issues.append(f"{lf.name}: parse error ({e}).")

            cdf = canonical_records_to_df(all_records)

            if all_geoms:
                geom_df = pd.concat(all_geoms, ignore_index=True)
                # De-duplicate geometries by (location_id, source_file)
                geom_df = geom_df.drop_duplicates(subset=["location_id", "source_file"], keep="first")
            else:
                geom_df = pd.DataFrame(columns=["location_id", "location_name", "geometry", "area_m2", "centroid_lat", "centroid_lon", "source_file"])

            st.session_state.canonical_df = cdf
            st.session_state.geom_df = geom_df

            # Inferred metric candidates for settings tab
            st.session_state.inferred_metrics = infer_metric_columns(cdf) if not cdf.empty else {}

            st.success(f"Parsing complete. Canonical records: {len(cdf):,}")
            if issues:
                st.warning("Some files could not be parsed fully:")
                for msg in issues:
                    st.write(f"- {msg}")

    # Display canonical preview
    cdf_now = st.session_state.canonical_df
    gdf_now = st.session_state.geom_df

    st.divider()
    st.subheader("Canonical data preview")
    if cdf_now is None or cdf_now.empty:
        st.info("No canonical data yet. Upload/select files and click Parse.")
    else:
        st.write("This is the internal canonical table used for all downstream analysis.")
        st.dataframe(cdf_now.head(200), use_container_width=True)

        with st.expander("Inferred schema hints (metric candidates)"):
            hints = st.session_state.inferred_metrics or {}
            st.json(hints)

    st.divider()
    st.subheader("Geometry preview (if available)")
    if gdf_now is None or gdf_now.empty:
        st.info("No geometry found yet (GeoJSON optional).")
    else:
        st.dataframe(gdf_now.head(200), use_container_width=True)


# -----------------------------
# Tab 2: Analysis Settings
# -----------------------------
with tab_settings:
    cdf_now = st.session_state.canonical_df
    if cdf_now is None or cdf_now.empty:
        st.info("Please parse some files first in the 'Upload & Detect' tab.")
    else:
        st.subheader("Time window and aggregation")
        ts_min = pd.to_datetime(cdf_now["timestamp"], errors="coerce").min()
        ts_max = pd.to_datetime(cdf_now["timestamp"], errors="coerce").max()

        has_time = pd.notna(ts_min) and pd.notna(ts_max)
        colA, colB, colC = st.columns([1, 1, 1], gap="large")

        with colA:
            if has_time:
                start_date = st.date_input("Start date", value=ts_min.date())
                end_date = st.date_input("End date", value=ts_max.date())
            else:
                st.info("No timestamps detected. Analysis will use static metrics where possible.")
                start_date = None
                end_date = None

        with colB:
            agg_method = st.selectbox("Aggregation for ΔT", ["Mean", "Median", "P95 (conservative)"], index=0)
            st.caption("This controls which representative ΔT is used for the central estimate.")

        with colC:
            min_points = st.number_input("Minimum points for time-series classification", min_value=1, max_value=50, value=3, step=1)

        st.divider()
        st.subheader("Baseline and metric selection")

        hints = st.session_state.inferred_metrics or {}
        surface_candidates = hints.get("surface_temp", [])
        ambient_candidates = hints.get("ambient_temp", [])
        anomaly_candidates = hints.get("anomaly", [])
        area_candidates = hints.get("area", [])
        hotspot_candidates = hints.get("hotspot", [])

        col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.0, 1.0], gap="large")

        with col1:
            surface_metric = st.selectbox(
                "Surface temperature metric (optional)",
                options=["(none)"] + sorted(set(surface_candidates)),
                index=0 if not surface_candidates else 1,
            )
        with col2:
            ambient_metric = st.selectbox(
                "Ambient temperature metric (optional)",
                options=["(none)"] + sorted(set(ambient_candidates)),
                index=0 if not ambient_candidates else 1,
            )
        with col3:
            anomaly_metric = st.selectbox(
                "Temperature anomaly metric (preferred if available)",
                options=["(none)"] + sorted(set(anomaly_candidates)),
                index=0 if not anomaly_candidates else 1,
            )
        with col4:
            area_metric = st.selectbox(
                "Area metric from data (optional)",
                options=["(none)"] + sorted(set(area_candidates)),
                index=0,
            )

        baseline_method = st.selectbox(
            "Ambient baseline method (if ambient metric is missing)",
            [
                "Manual constant ambient (°C)",
                "Use per-location median surface temperature (relative baseline)",
                "Use per-location P10 surface temperature (relative baseline)",
            ],
            index=0,
        )

        manual_ambient_c = st.number_input("Manual ambient (°C)", value=10.0, step=0.5)

        st.divider()
        st.subheader("Heat-loss model parameters (screening-grade)")

        colP1, colP2, colP3, colP4 = st.columns([1, 1, 1, 1], gap="large")

        with colP1:
            h_conv = st.number_input("Convective h (W/m²K)", value=10.0, min_value=1.0, max_value=40.0, step=0.5)
            emissivity = st.number_input("Surface emissivity ε", value=0.90, min_value=0.50, max_value=1.00, step=0.01)

        with colP2:
            default_area_m2 = st.number_input("Default area if unknown (m²)", value=2000.0, min_value=10.0, step=50.0)
            st.caption("Used when no geometry or area metric is available.")

        with colP3:
            scaling_method = st.selectbox(
                "Annual scaling method",
                ["Seasonal fraction", "Fixed runtime hours", "Degree-day (HDD)"],
                index=0,
            )
            season_fraction = st.slider("Heating season fraction (0 to 1)", min_value=0.0, max_value=1.0, value=0.50, step=0.05)

        with colP4:
            hours_per_year = st.number_input("Runtime hours per year (if selected)", value=3500.0, min_value=0.0, max_value=8760.0, step=100.0)
            annual_hdd_kdays = st.number_input("Annual HDD (K·days) (if selected)", value=2500.0, min_value=0.0, max_value=8000.0, step=100.0)

        st.divider()
        st.subheader("Tariffs and carbon factors")
        colT1, colT2, colT3 = st.columns([1, 1, 1], gap="large")
        with colT1:
            tariff_gbp_per_kwh = st.number_input("Tariff (£/kWh)", value=0.15, min_value=0.0, max_value=2.0, step=0.01)
        with colT2:
            carbon_kg_per_kwh = st.number_input("Carbon factor (kgCO₂e/kWh)", value=0.20, min_value=0.0, max_value=2.0, step=0.01)
        with colT3:
            avoidable_fraction = st.slider("Avoidable fraction (screening)", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
            st.caption("Represents the portion of losses that could realistically be avoided by interventions.")

        st.divider()
        st.subheader("Uncertainty (Monte Carlo bounds)")
        colU1, colU2, colU3, colU4 = st.columns([1, 1, 1, 1], gap="large")
        with colU1:
            n_mc = st.number_input("Monte Carlo samples", value=800, min_value=200, max_value=5000, step=100)
        with colU2:
            h_conv_rel_range = st.slider("h uncertainty (relative)", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
            emissivity_abs_range = st.slider("ε uncertainty (absolute)", min_value=0.0, max_value=0.30, value=0.05, step=0.01)
        with colU3:
            area_rel_range = st.slider("Area uncertainty (relative)", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
            scaling_rel_range = st.slider("Annual scaling uncertainty (relative)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        with colU4:
            ambient_abs_range_c = st.slider("Ambient uncertainty (°C)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            dt_abs_range_c = st.slider("ΔT measurement uncertainty (°C)", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

        st.divider()
        run_analysis = st.button("Run analysis", type="primary")

        # Store assumptions in session state
        st.session_state.assumptions = {
            "time_window": {"start": str(start_date) if start_date else None, "end": str(end_date) if end_date else None},
            "agg_method": agg_method,
            "metric_selection": {
                "surface_metric": None if surface_metric == "(none)" else surface_metric,
                "ambient_metric": None if ambient_metric == "(none)" else ambient_metric,
                "anomaly_metric": None if anomaly_metric == "(none)" else anomaly_metric,
                "area_metric": None if area_metric == "(none)" else area_metric,
                "hotspot_candidates": hotspot_candidates[:10],
            },
            "baseline_method": baseline_method,
            "manual_ambient_c": float(manual_ambient_c),
            "heat_model": {
                "h_conv": float(h_conv),
                "emissivity": float(emissivity),
                "default_area_m2": float(default_area_m2),
                "scaling_method": scaling_method,
                "season_fraction": float(season_fraction),
                "hours_per_year": float(hours_per_year),
                "annual_hdd_kdays": float(annual_hdd_kdays),
            },
            "economics": {
                "tariff_gbp_per_kwh": float(tariff_gbp_per_kwh),
                "carbon_kg_per_kwh": float(carbon_kg_per_kwh),
                "avoidable_fraction": float(avoidable_fraction),
            },
            "uncertainty": {
                "n_mc": int(n_mc),
                "h_conv_rel_range": float(h_conv_rel_range),
                "emissivity_abs_range": float(emissivity_abs_range),
                "area_rel_range": float(area_rel_range),
                "ambient_abs_range_c": float(ambient_abs_range_c),
                "dt_abs_range_c": float(dt_abs_range_c),
                "scaling_rel_range": float(scaling_rel_range),
            },
        }

        if run_analysis:
            # Perform analysis now
            cdf = st.session_state.canonical_df.copy()
            gdf = st.session_state.geom_df.copy()

            # Time filter
            if has_time and start_date and end_date:
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                cdf_f = cdf[(cdf["timestamp"].isna()) | ((cdf["timestamp"] >= start_ts) & (cdf["timestamp"] <= end_ts))].copy()
            else:
                cdf_f = cdf.copy()

            # Metric picks
            surface_pick = None if surface_metric == "(none)" else surface_metric
            ambient_pick = None if ambient_metric == "(none)" else ambient_metric
            anomaly_pick = None if anomaly_metric == "(none)" else anomaly_metric
            area_pick = None if area_metric == "(none)" else area_metric

            # Build per-location series for dt and ambient
            # Strategy:
            # 1) If anomaly metric exists: dt = anomaly
            # 2) Else if surface + ambient: dt = surface - ambient
            # 3) Else if surface only: dt = surface - baseline(surface) using baseline method
            # 4) Else: dt = 0 (cannot estimate)
            def get_metric_series(loc_id: str, metric: str) -> pd.DataFrame:
                dd = cdf_f[(cdf_f["location_id"] == loc_id) & (cdf_f["metric_name"] == metric)].copy()
                dd = dd.sort_values("timestamp")
                return dd

            locations = sorted(set(cdf_f["location_id"].dropna().astype(str).tolist()))
            if not locations:
                st.error("No locations detected in canonical data.")
                st.stop()

            rng = np.random.default_rng(42)

            results_rows = []
            per_location_timeseries = {}

            progress = st.progress(0)
            for idx, loc_id in enumerate(locations):
                loc_df = cdf_f[cdf_f["location_id"] == loc_id].copy()
                loc_name = loc_df["location_name"].dropna().astype(str).iloc[0] if loc_df["location_name"].notna().any() else str(loc_id)

                # Geometry area if present
                area_geo = None
                centroid_lat = None
                centroid_lon = None
                if gdf is not None and not gdf.empty:
                    gg = gdf[gdf["location_id"].astype(str) == str(loc_id)]
                    if not gg.empty:
                        area_geo = gg["area_m2"].dropna().astype(float).iloc[0] if gg["area_m2"].notna().any() else None
                        centroid_lat = gg["centroid_lat"].dropna().astype(float).iloc[0] if gg["centroid_lat"].notna().any() else None
                        centroid_lon = gg["centroid_lon"].dropna().astype(float).iloc[0] if gg["centroid_lon"].notna().any() else None

                # Area from metric (if any)
                area_metric_val = None
                if area_pick:
                    av = loc_df[loc_df["metric_name"] == area_pick]["metric_value"]
                    if av.notna().any():
                        area_metric_val = float(av.dropna().iloc[-1])

                area_m2 = area_geo if area_geo and np.isfinite(area_geo) and area_geo > 0 else None
                if area_m2 is None and area_metric_val and np.isfinite(area_metric_val) and area_metric_val > 0:
                    area_m2 = area_metric_val
                if area_m2 is None:
                    area_m2 = float(default_area_m2)

                # Series extraction
                anomaly_series = None
                surface_series = None
                ambient_series = None

                if anomaly_pick:
                    a = get_metric_series(loc_id, anomaly_pick)
                    if not a.empty:
                        anomaly_series = a

                if surface_pick:
                    s = get_metric_series(loc_id, surface_pick)
                    if not s.empty:
                        surface_series = s

                if ambient_pick:
                    a2 = get_metric_series(loc_id, ambient_pick)
                    if not a2.empty:
                        ambient_series = a2

                # Build dt time series
                ts_df = None
                if anomaly_series is not None and not anomaly_series.empty:
                    ts_df = anomaly_series[["timestamp", "metric_value"]].rename(columns={"metric_value": "dt"})
                elif surface_series is not None and not surface_series.empty and ambient_series is not None and not ambient_series.empty:
                    # Align by timestamp if possible (nearest merge)
                    ss = surface_series[["timestamp", "metric_value"]].rename(columns={"metric_value": "surface"})
                    aa = ambient_series[["timestamp", "metric_value"]].rename(columns={"metric_value": "ambient"})
                    if ss["timestamp"].notna().any() and aa["timestamp"].notna().any():
                        ss2 = ss.dropna(subset=["timestamp"]).sort_values("timestamp")
                        aa2 = aa.dropna(subset=["timestamp"]).sort_values("timestamp")
                        merged = pd.merge_asof(ss2, aa2, on="timestamp", direction="nearest", tolerance=pd.Timedelta("2H"))
                        merged["dt"] = merged["surface"] - merged["ambient"]
                        ts_df = merged[["timestamp", "dt", "ambient", "surface"]]
                    else:
                        # Static
                        dt_static = float(np.nanmean(ss["surface"].values) - np.nanmean(aa["ambient"].values))
                        ts_df = pd.DataFrame({"timestamp": [pd.NaT], "dt": [dt_static]})
                elif surface_series is not None and not surface_series.empty:
                    ss = surface_series[["timestamp", "metric_value"]].rename(columns={"metric_value": "surface"})
                    # Baseline from surface itself
                    surf_vals = pd.to_numeric(ss["surface"], errors="coerce").dropna().values.astype(float)
                    if surf_vals.size == 0:
                        ts_df = pd.DataFrame({"timestamp": [pd.NaT], "dt": [0.0]})
                    else:
                        if baseline_method == "Use per-location median surface temperature (relative baseline)":
                            base = float(np.nanmedian(surf_vals))
                        elif baseline_method == "Use per-location P10 surface temperature (relative baseline)":
                            base = float(np.nanpercentile(surf_vals, 10))
                        else:
                            base = float(manual_ambient_c)
                        ss["ambient"] = base
                        ss["dt"] = ss["surface"] - base
                        ts_df = ss[["timestamp", "dt", "ambient", "surface"]]
                else:
                    ts_df = pd.DataFrame({"timestamp": [pd.NaT], "dt": [0.0]})

                # Ambient series for uncertainty
                if "ambient" in ts_df.columns and ts_df["ambient"].notna().any():
                    ambient_vals = pd.to_numeric(ts_df["ambient"], errors="coerce").dropna().values.astype(float)
                else:
                    ambient_vals = np.array([float(manual_ambient_c)])

                dt_vals = pd.to_numeric(ts_df["dt"], errors="coerce").dropna().values.astype(float)
                dt_vals = np.maximum(dt_vals, 0.0)

                # Central aggregation for dt
                if dt_vals.size == 0:
                    dt_central = 0.0
                else:
                    if agg_method == "Median":
                        dt_central = float(np.nanmedian(dt_vals))
                    elif agg_method.startswith("P95"):
                        dt_central = float(np.nanpercentile(dt_vals, 95))
                    else:
                        dt_central = float(np.nanmean(dt_vals))

                # Central ambient
                amb_central = float(np.nanmean(ambient_vals)) if ambient_vals.size else float(manual_ambient_c)

                # Central flux + annual energy
                q_central = float(heat_flux_w_m2(np.array([dt_central]), np.array([amb_central]), float(h_conv), float(emissivity))[0])
                e_kwh_central = float(
                    annualise_energy_kwh(
                        q_w_m2=np.array([q_central]),
                        area_m2=float(area_m2),
                        method=scaling_method,
                        hours_per_year=float(hours_per_year),
                        season_fraction=float(season_fraction),
                        annual_hdd_kdays=float(annual_hdd_kdays),
                        ambient_c=np.array([amb_central]),
                        delta_t_c=np.array([dt_central]),
                        h_conv=float(h_conv),
                        emissivity=float(emissivity),
                    )[0]
                )

                # Hotspot ratio proxy from any "max/mean" if present
                hotspot_ratio = None
                # Compute from dt distribution peakiness
                if dt_vals.size >= 3:
                    hotspot_ratio = float(np.nanpercentile(dt_vals, 95) / max(np.nanpercentile(dt_vals, 50), 1e-6))

                # Classification
                data_richness = int(dt_vals.size)
                label, reason = classify_location(dt_vals, hotspot_ratio, data_richness)

                # Confidence scoring (simple)
                confidence = "Low"
                if anomaly_pick or (surface_pick and ambient_pick):
                    confidence = "Medium"
                if (area_geo is not None and np.isfinite(area_geo) and area_geo > 0) and (anomaly_pick or (surface_pick and ambient_pick)):
                    confidence = "High"

                # Uncertainty bounds
                bounds = compute_uncertainty_bounds(
                    dt_samples=dt_vals if dt_vals.size else np.array([dt_central]),
                    ambient_samples=ambient_vals,
                    area_m2=float(area_m2),
                    tariff_gbp_per_kwh=float(tariff_gbp_per_kwh),
                    carbon_kg_per_kwh=float(carbon_kg_per_kwh),
                    method=scaling_method,
                    hours_per_year=float(hours_per_year),
                    season_fraction=float(season_fraction),
                    annual_hdd_kdays=float(annual_hdd_kdays),
                    n_mc=int(n_mc),
                    h_conv_central=float(h_conv),
                    h_conv_rel_range=float(h_conv_rel_range),
                    emissivity_central=float(emissivity),
                    emissivity_abs_range=float(emissivity_abs_range),
                    area_rel_range=float(area_rel_range),
                    ambient_abs_range_c=float(ambient_abs_range_c),
                    dt_abs_range_c=float(dt_abs_range_c),
                    scaling_rel_range=float(scaling_rel_range),
                    rng=rng,
                )

                # Avoidable portion
                e_kwh_avoidable_p50 = bounds["energy_kwh"]["p50"] * float(avoidable_fraction)
                cost_avoidable_p50 = bounds["cost_gbp"]["p50"] * float(avoidable_fraction)
                carbon_avoidable_p50 = bounds["carbon_tco2e"]["p50"] * float(avoidable_fraction)

                results_rows.append(
                    {
                        "location_id": str(loc_id),
                        "location_name": str(loc_name),
                        "area_m2_used": float(area_m2),
                        "dt_central_c": float(dt_central),
                        "q_central_w_m2": float(q_central),
                        "annual_kwh_p50": float(bounds["energy_kwh"]["p50"]),
                        "annual_kwh_p10": float(bounds["energy_kwh"]["p10"]),
                        "annual_kwh_p90": float(bounds["energy_kwh"]["p90"]),
                        "annual_cost_gbp_p50": float(bounds["cost_gbp"]["p50"]),
                        "annual_cost_gbp_p10": float(bounds["cost_gbp"]["p10"]),
                        "annual_cost_gbp_p90": float(bounds["cost_gbp"]["p90"]),
                        "annual_carbon_tco2e_p50": float(bounds["carbon_tco2e"]["p50"]),
                        "annual_carbon_tco2e_p10": float(bounds["carbon_tco2e"]["p10"]),
                        "annual_carbon_tco2e_p90": float(bounds["carbon_tco2e"]["p90"]),
                        "avoidable_kwh_p50": float(e_kwh_avoidable_p50),
                        "avoidable_cost_gbp_p50": float(cost_avoidable_p50),
                        "avoidable_carbon_tco2e_p50": float(carbon_avoidable_p50),
                        "classification": label,
                        "classification_reason": reason,
                        "confidence": confidence,
                        "n_points": int(data_richness),
                        "centroid_lat": float(centroid_lat) if centroid_lat is not None and np.isfinite(centroid_lat) else None,
                        "centroid_lon": float(centroid_lon) if centroid_lon is not None and np.isfinite(centroid_lon) else None,
                    }
                )

                # Store time series for drill-down (small)
                per_location_timeseries[str(loc_id)] = ts_df.copy()

                progress.progress(int((idx + 1) / max(1, len(locations)) * 100))

            progress.empty()

            res_df = pd.DataFrame(results_rows)
            # Rank by avoidable cost
            res_df = res_df.sort_values("avoidable_cost_gbp_p50", ascending=False).reset_index(drop=True)

            st.session_state.analysis_results = {
                "results_table": res_df,
                "timeseries": per_location_timeseries,
            }

            st.success("Analysis completed. View results in the 'Results' tab.")


# -----------------------------
# Tab 3: Results
# -----------------------------
with tab_results:
    analysis = st.session_state.analysis_results
    if analysis is None:
        st.info("Run the analysis first in the 'Analysis Settings' tab.")
    else:
        res_df: pd.DataFrame = analysis["results_table"]
        ts_map: Dict[str, pd.DataFrame] = analysis["timeseries"]

        st.subheader("Ranked results (screening-grade)")

        # Summary stats cards
        c1, c2, c3, c4 = st.columns(4, gap="large")
        with c1:
            st.metric("Locations analysed", f"{len(res_df):,}")
        with c2:
            st.metric("Total avoidable cost (P50)", f"£{res_df['avoidable_cost_gbp_p50'].sum():,.0f}")
        with c3:
            st.metric("Total avoidable energy (P50)", f"{res_df['avoidable_kwh_p50'].sum():,.0f} kWh")
        with c4:
            st.metric("Total avoidable carbon (P50)", f"{res_df['avoidable_carbon_tco2e_p50'].sum():,.1f} tCO₂e")

        st.dataframe(res_df, use_container_width=True, height=360)

        st.divider()
        st.subheader("Distribution across locations")
        # Simple histogram via pandas (Streamlit native chart)
        dist_col = st.selectbox("Choose metric for distribution", ["avoidable_cost_gbp_p50", "avoidable_kwh_p50", "annual_kwh_p50"], index=0)
        st.bar_chart(res_df.set_index("location_id")[dist_col], height=220)

        st.divider()
        st.subheader("Per-location drill-down")

        sel_id = st.selectbox("Select location_id", options=res_df["location_id"].tolist(), index=0)
        sel_row = res_df[res_df["location_id"] == sel_id].iloc[0].to_dict()

        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown(f"**{sel_row['location_name']}** (`{sel_id}`)")
            st.write(
                {
                    "Classification": sel_row["classification"],
                    "Reason": sel_row["classification_reason"],
                    "Confidence": sel_row["confidence"],
                    "Area used (m²)": sel_row["area_m2_used"],
                    "Central ΔT (°C)": round(sel_row["dt_central_c"], 2),
                    "Central q (W/m²)": round(sel_row["q_central_w_m2"], 2),
                }
            )

            st.write("Annual impact (P10 / P50 / P90)")
            st.write(
                {
                    "Energy (kWh)": (round(sel_row["annual_kwh_p10"], 0), round(sel_row["annual_kwh_p50"], 0), round(sel_row["annual_kwh_p90"], 0)),
                    "Cost (£)": (round(sel_row["annual_cost_gbp_p10"], 0), round(sel_row["annual_cost_gbp_p50"], 0), round(sel_row["annual_cost_gbp_p90"], 0)),
                    "Carbon (tCO₂e)": (round(sel_row["annual_carbon_tco2e_p10"], 2), round(sel_row["annual_carbon_tco2e_p50"], 2), round(sel_row["annual_carbon_tco2e_p90"], 2)),
                }
            )

            st.write("Avoidable portion (P50)")
            st.write(
                {
                    "Avoidable energy (kWh)": round(sel_row["avoidable_kwh_p50"], 0),
                    "Avoidable cost (£)": round(sel_row["avoidable_cost_gbp_p50"], 0),
                    "Avoidable carbon (tCO₂e)": round(sel_row["avoidable_carbon_tco2e_p50"], 2),
                }
            )

        with right:
            st.markdown("**Time series (if available)**")
            ts_df = ts_map.get(str(sel_id))
            if ts_df is None or ts_df.empty or ts_df["timestamp"].isna().all():
                st.info("No timestamps available for this location. Showing static summary only.")
            else:
                # Clean and chart
                ts_plot = ts_df.copy()
                ts_plot["timestamp"] = pd.to_datetime(ts_plot["timestamp"], errors="coerce")
                ts_plot = ts_plot.dropna(subset=["timestamp"])
                if "dt" in ts_plot.columns:
                    st.line_chart(ts_plot.set_index("timestamp")[["dt"]], height=220)
                if "surface" in ts_plot.columns and "ambient" in ts_plot.columns:
                    st.line_chart(ts_plot.set_index("timestamp")[["surface", "ambient"]], height=220)

        st.divider()
        st.subheader("Map (if geometry/centroids exist)")
        # Build a map table from results
        map_df = res_df.copy()
        map_df = map_df.dropna(subset=["centroid_lat", "centroid_lon"], how="any")
        if map_df.empty:
            st.info("No centroid coordinates available. Upload GeoJSON footprints (optional) to enable mapping.")
        else:
            # Use pydeck scatterplot
            import pydeck as pdk

            map_df = map_df.copy()
            # Normalise size
            v = map_df["avoidable_cost_gbp_p50"].values.astype(float)
            v = np.where(np.isfinite(v), v, 0.0)
            v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
            map_df["radius"] = 200 + 1400 * v_norm

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["centroid_lon", "centroid_lat"],
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )

            view_state = pdk.ViewState(
                latitude=float(map_df["centroid_lat"].mean()),
                longitude=float(map_df["centroid_lon"].mean()),
                zoom=14,
                pitch=0,
            )

            tooltip = {
                "html": "<b>{location_name}</b><br/>Avoidable cost (P50): £{avoidable_cost_gbp_p50}<br/>Avoidable kWh (P50): {avoidable_kwh_p50}",
                "style": {"backgroundColor": "white", "color": "black"},
            }

            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))


# -----------------------------
# Tab 4: Export
# -----------------------------
with tab_export:
    analysis = st.session_state.analysis_results
    assumptions = st.session_state.assumptions or {}

    st.subheader("Export outputs")
    if analysis is None:
        st.info("Run the analysis first to enable exports.")
    else:
        res_df: pd.DataFrame = analysis["results_table"]
        ts_map: Dict[str, pd.DataFrame] = analysis["timeseries"]

        colE1, colE2, colE3 = st.columns([1, 1, 1], gap="large")

        with colE1:
            csv_bytes = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name="thermal_to_capex_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with colE2:
            # JSON export includes assumptions and small per-location time series (if present)
            export_obj = {
                "tool": APP_TITLE,
                "generated_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "assumptions": assumptions,
                "results_table": res_df.to_dict(orient="records"),
                "time_series": {},
            }
            # Keep time series modest in size
            for loc_id, df in ts_map.items():
                if df is None or df.empty:
                    continue
                dd = df.copy()
                if "timestamp" in dd.columns:
                    dd["timestamp"] = dd["timestamp"].astype(str)
                export_obj["time_series"][str(loc_id)] = dd.head(2000).to_dict(orient="records")

            json_bytes = json.dumps(export_obj, indent=2).encode("utf-8")
            st.download_button(
                "Download results JSON",
                data=json_bytes,
                file_name="thermal_to_capex_results.json",
                mime="application/json",
                use_container_width=True,
            )

        with colE3:
            html = build_html_report(APP_TITLE, res_df, assumptions)
            st.download_button(
                "Download simple HTML report",
                data=html.encode("utf-8"),
                file_name="thermal_to_capex_report.html",
                mime="text/html",
                use_container_width=True,
            )

        st.divider()
        st.subheader("Assumptions (for audit trail)")
        st.json(assumptions)

        st.divider()
        st.subheader("requirements.txt (copy-paste)")
        REQUIREMENTS_TXT = """streamlit>=1.36,<2
pandas>=2.0,<3
numpy>=1.24,<3
pydeck>=0.8,<1
shapely>=2.0,<3
pyproj>=3.6,<4
"""
        st.code(REQUIREMENTS_TXT.strip(), language="text")


# -----------------------------
# HTML report builder
# -----------------------------
def build_html_report(title: str, res_df: pd.DataFrame, assumptions: Dict[str, Any]) -> str:
    """
    Simple offline HTML report: summary + ranked table + assumptions.
    No external assets.
    """
    # Keep the table manageable
    top = res_df.copy().head(50)

    def esc(x: Any) -> str:
        s = "" if x is None else str(x)
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    rows = []
    for _, r in top.iterrows():
        rows.append(
            "<tr>"
            f"<td>{esc(r.get('location_id'))}</td>"
            f"<td>{esc(r.get('location_name'))}</td>"
            f"<td style='text-align:right'>{float(r.get('area_m2_used', 0.0)):.0f}</td>"
            f"<td style='text-align:right'>{float(r.get('dt_central_c', 0.0)):.2f}</td>"
            f"<td style='text-align:right'>£{float(r.get('avoidable_cost_gbp_p50', 0.0)):.0f}</td>"
            f"<td style='text-align:right'>{float(r.get('avoidable_kwh_p50', 0.0)):.0f}</td>"
            f"<td style='text-align:right'>{float(r.get('avoidable_carbon_tco2e_p50', 0.0)):.2f}</td>"
            f"<td>{esc(r.get('classification'))}</td>"
            f"<td>{esc(r.get('confidence'))}</td>"
            "</tr>"
        )

    total_avoid_cost = float(res_df["avoidable_cost_gbp_p50"].sum()) if "avoidable_cost_gbp_p50" in res_df.columns else 0.0
    total_avoid_kwh = float(res_df["avoidable_kwh_p50"].sum()) if "avoidable_kwh_p50" in res_df.columns else 0.0
    total_avoid_t = float(res_df["avoidable_carbon_tco2e_p50"].sum()) if "avoidable_carbon_tco2e_p50" in res_df.columns else 0.0

    assumptions_json = esc(json.dumps(assumptions, indent=2))

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{esc(title)} Report</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #111; }}
h1 {{ margin: 0 0 8px 0; }}
p {{ margin: 6px 0; }}
.small {{ font-size: 12px; color: #444; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 14px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; }}
th {{ background: #f5f5f5; text-align: left; }}
tr:nth-child(even) {{ background: #fafafa; }}
.badge {{ display:inline-block; padding: 2px 8px; border-radius: 10px; background:#eee; font-size: 12px; }}
pre {{ background:#f7f7f7; padding: 12px; overflow:auto; }}
</style>
</head>
<body>
<h1>{esc(title)}: Screening Report</h1>
<p class="small">Generated (UTC): {esc(_dt.datetime.utcnow().isoformat(timespec="seconds"))}Z</p>

<h2>Executive summary</h2>
<p>This report provides screening-grade estimates of outward heat loss, annual energy loss, cost, and carbon impact based on available Earth observation evidence and simplified physics.</p>
<ul>
  <li><b>Total avoidable cost (P50):</b> £{total_avoid_cost:,.0f}</li>
  <li><b>Total avoidable energy (P50):</b> {total_avoid_kwh:,.0f} kWh</li>
  <li><b>Total avoidable carbon (P50):</b> {total_avoid_t:,.2f} tCO₂e</li>
</ul>

<h2>Top locations by avoidable cost (P50)</h2>
<table>
  <thead>
    <tr>
      <th>Location ID</th>
      <th>Name</th>
      <th>Area used (m²)</th>
      <th>Central ΔT (°C)</th>
      <th>Avoidable cost (£/yr)</th>
      <th>Avoidable energy (kWh/yr)</th>
      <th>Avoidable carbon (tCO₂e/yr)</th>
      <th>Classification</th>
      <th>Confidence</th>
    </tr>
  </thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>

<h2>Assumptions (audit trail)</h2>
<pre>{assumptions_json}</pre>

<p class="small">Important: This is a screening tool. Results should be validated with site surveys, metered data, and engineering review before any investment decision.</p>
</body>
</html>
"""
    return html
