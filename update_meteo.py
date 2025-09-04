# -*- coding: utf-8 -*-
"""
update_meteo.py — Autoupdate histórico a partir del pronóstico MeteoBahía.
- Lee/parsea el XML (robusto a <tag value="..."> o texto).
- Fusiona con histórico existente (repo y/o local).
- Asegura continuidad diaria hasta el horizonte (hoy + N).
- Guarda en data/meteo_daily.csv y (opcional) en meteo_history.csv para la app.
"""

import os
import sys
import pytz
import datetime as dt
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path

# ================== Config ==================
API_URL         = os.getenv("METEOBAHIA_URL", "https://meteobahia.com.ar/scripts/forecast/for-bd.xml")
PRON_DIAS_API   = int(os.getenv("PRON_DIAS_API", "8"))   # hoy + 7
TZ              = pytz.timezone(os.getenv("TIMEZONE", "America/Argentina/Buenos_Aires"))

# Salidas (repo y app)
CSV_REPO_PATH   = os.getenv("GH_PATH", "data/meteo_daily.csv")  # usado por workflows/analítica
CSV_APP_PATH    = os.getenv("APP_HISTORY_PATH", "meteo_history.csv")  # leído por la app si querés

# ================== Utils ==================
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def today_local():
    return dt.datetime.now(TZ).date()

def _get_attr_or_text(elem, attr="value"):
    if elem is None:
        return None
    # primero intento atributo (p.ej. <tmax value="25.1"/>)
    v = (elem.attrib or {}).get(attr)
    if v is not None and str(v).strip():
        return str(v).strip()
    # si no, uso el texto interno (<tmax>25.1</tmax>)
    t = (elem.text or "").strip()
    return t if t else None

def parse_api_xml(url: str):
    """Devuelve lista de dicts con date, tmax, tmin, prec, source='forecast'."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        xml_bytes = r.read()
    root = ET.fromstring(xml_bytes)

    out = []
    # Soportar estructura con <day> y subtags con atributo value="..."
    for day in root.iter():
        if day.tag.lower().endswith("day"):
            # Buscar sub-tags típicos
            fecha = None
            tmax = tmin = prec = None

            # nombres usuales
            fecha_tag  = day.find("fecha")
            tmax_tag   = day.find("tmax")
            tmin_tag   = day.find("tmin")
            precip_tag = day.find("precip")

            if fecha_tag is not None:
                fecha = _get_attr_or_text(fecha_tag, "value")

            if tmax_tag is not None:
                tmax_s = _get_attr_or_text(tmax_tag, "value")
                if tmax_s is not None:
                    try: tmax = float(str(tmax_s).replace(",", "."))
                    except: pass

            if tmin_tag is not None:
                tmin_s = _get_attr_or_text(tmin_tag, "value")
                if tmin_s is not None:
                    try: tmin = float(str(tmin_s).replace(",", "."))
                    except: pass

            if precip_tag is not None:
                prec_s = _get_attr_or_text(precip_tag, "value")
                if prec_s is not None:
                    try: prec = float(str(prec_s).replace(",", "."))
                    except: pass

            # fallback genérico si no estaban esos nombres
            if fecha is None:
                for child in day:
                    tag = child.tag.lower()
                    val = _get_attr_or_text(child, "value")
                    if "date" in tag or "fecha" in tag:
                        fecha = fecha or val
                    elif "tmax" in tag or ("max" in tag and "t" in tag):
                        if tmax is None and val is not None:
                            try: tmax = float(val.replace(",", "."))
                            except: pass
                    elif "tmin" in tag or ("min" in tag and "t" in tag):
                        if tmin is None and val is not None:
                            try: tmin = float(val.replace(",", "."))
                            except: pass
                    elif "prec" in tag or "rain" in tag or "pp" in tag:
                        if prec is None and val is not None:
                            try: prec = float(val.replace(",", "."))
                            except: pass

            # Normalizar fecha
            d = None
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
                if fecha:
                    try:
                        d = dt.datetime.strptime(fecha, fmt).date()
                        break
                    except:
                        continue

            if d is None or tmax is None or tmin is None:
                continue
            if prec is None:
                prec = 0.0

            out.append({"date": d, "tmax": tmax, "tmin": tmin, "prec": prec, "source": "forecast"})
    return out

def load_existing(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["date","jd","tmax","tmin","prec","source","updated_at"])
    df = pd.read_csv(csv_path, parse_dates=["date","updated_at"], dayfirst=False)
    df["date"] = df["date"].dt.date
    return df

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(path)
    df2 = df.sort_values("date").copy()
    df2.to_csv(path, index=False)

# ================== Main ==================
def main():
    today = today_local()
    horizon_end = today + dt.timedelta(days=PRON_DIAS_API - 1)

    # Cargar históricos existentes (repo y app si existieran) y unirlos
    df_repo = load_existing(CSV_REPO_PATH)
    df_app  = load_existing(CSV_APP_PATH)
    if df_repo.empty and df_app.empty:
        df_hist = pd.DataFrame(columns=["date","jd","tmax","tmin","prec","source","updated_at"])
    else:
        both = pd.concat([df_repo, df_app], ignore_index=True)
        both = (both.drop_duplicates(subset=["date"], keep="last")
                    .sort_values("date")
                    .reset_index(drop=True))
        df_hist = both

    # Leer pronóstico
    try:
        forecast = parse_api_xml(API_URL)
    except (HTTPError, URLError, ET.ParseError) as e:
        print(f"[WARN] No se pudo leer la API: {e}", file=sys.stderr)
        forecast = []

    fdf = pd.DataFrame(forecast)
    if not fdf.empty:
        fdf = fdf[(fdf["date"] >= today) & (fdf["date"] <= horizon_end)]
        fdf["jd"] = fdf["date"].apply(lambda d: d.timetuple().tm_yday)
        fdf["updated_at"] = pd.Timestamp.now(TZ)
    else:
        # Sin datos de API: igual mantenemos el horizonte para no cortar la serie
        dates = pd.date_range(today, horizon_end, freq="D").date
        fdf = pd.DataFrame({
            "date": dates,
            "tmax": np.nan, "tmin": np.nan, "prec": np.nan,
            "source": "forecast",
            "jd": [d.timetuple().tm_yday for d in dates],
            "updated_at": pd.Timestamp.now(TZ)
        })

    # Fusionar: reemplazar ventana [today, horizon_end] con la API más reciente
    if df_hist.empty:
        merged = fdf.copy()
    else:
        lo, hi = fdf["date"].min(), fdf["date"].max()
        mask = (df_hist["date"] >= lo) & (df_hist["date"] <= hi)
        merged = pd.concat([df_hist.loc[~mask].copy(), fdf.copy()], ignore_index=True)

    # Asegurar continuidad desde el mínimo disponible hasta horizon_end
    min_date = merged["date"].min() if not merged.empty else today
    full_idx = pd.date_range(min_date, horizon_end, freq="D").date
    merged = (merged.set_index("date")
                    .reindex(full_idx)
                    .reset_index()
                    .rename(columns={"index":"date"}))

    # Completar columnas clave
    merged["jd"]         = merged["date"].apply(lambda d: d.timetuple().tm_yday)
    merged["updated_at"] = merged["updated_at"].fillna(pd.Timestamp.now(TZ))
    # Imputación conservadora (sin dejar NaN)
    merged["tmax"]   = pd.to_numeric(merged["tmax"], errors="coerce").fillna(method="ffill")
    merged["tmin"]   = pd.to_numeric(merged["tmin"], errors="coerce").fillna(method="ffill")
    merged["prec"]   = pd.to_numeric(merged["prec"], errors="coerce").fillna(0.0)
    merged["source"] = merged["source"].fillna("forecast")

    # Guardar en ambas rutas (repo + app)
    save_csv(merged, CSV_REPO_PATH)
    if CSV_APP_PATH:
        save_csv(merged, CSV_APP_PATH)

    print(f"[OK] Histórico actualizado hasta {horizon_end} | filas={len(merged)} "
          f"| rango={merged['date'].min()} → {merged['date'].max()}")

if __name__ == "__main__":
    main()
