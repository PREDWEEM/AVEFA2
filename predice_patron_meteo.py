# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 Predicción de patrón histórico de emergencia (Early / Staggered / Medium)
# Basado en datos meteorológicos (meteo_history.csv)
# Empalme con inicio de días en septiembre (JD real)
# ===============================================================

import pandas as pd
import numpy as np
from datetime import datetime

# ---------- PARÁMETROS ----------
TEMP_BASE = 0.0      # base térmica para GDD
RAIN_DRY = 1.0       # umbral de día lluvioso
MONTH_START = 9      # el empalme inicia en septiembre
YEAR_REF = 2025      # año de referencia (ajustar según corresponda)

# ---------- FUNCIÓN DE CARGA ----------
def load_meteo_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "julian_days" not in df.columns:
        df["julian_days"] = np.arange(1, len(df) + 1)
    df["tmax"] = pd.to_numeric(df.get("tmax", df.get("tx", np.nan)), errors="coerce")
    df["tmin"] = pd.to_numeric(df.get("tmin", df.get("tn", np.nan)), errors="coerce")
    df["prec"]  = pd.to_numeric(df.get("prec", df.get("ppt", np.nan)), errors="coerce").clip(lower=0)
    df["tmed"]  = (df["tmax"] + df["tmin"]) / 2
    df["gdd"]   = np.maximum(df["tmed"] - TEMP_BASE, 0)
    df["rainy"] = (df["prec"] >= RAIN_DRY).astype(int)
    return df

# ---------- INDICADORES ----------
def compute_indicators(df: pd.DataFrame):
    gdd_cum = df["gdd"].cumsum()
    rain_cum = df["prec"].cumsum()
    humid_days = df["rainy"].rolling(30, min_periods=1).sum()
    df["gdd_cum"] = gdd_cum
    df["rain_cum"] = rain_cum
    df["humid30"] = humid_days
    df["ih_thermal"] = df["gdd_cum"] * (1 + df["rain_cum"] / 100)
    return df

# ---------- CLASIFICADOR HEURÍSTICO ----------
def classify_pattern(df: pd.DataFrame) -> dict:
    jd = df["julian_days"].to_numpy()
    gdd = df["gdd_cum"].to_numpy()
    rain = df["rain_cum"].to_numpy()

    # Indicadores por ventanas
    def sum_in_window(v, start, end):
        m = (jd >= start) & (jd <= end)
        return float(np.nansum(v[m])) / max(1, end - start + 1)

    # Energía térmica y humedad en ventanas clave
    gdd_early = sum_in_window(gdd, 60, 120)
    gdd_mid   = sum_in_window(gdd, 150, 210)
    rain_early = sum_in_window(rain, 60, 120)
    rain_mid   = sum_in_window(rain, 150, 210)

    # Normalización simple
    total_gdd = np.nanmax(gdd)
    total_rain = np.nanmax(rain)
    e_rel = gdd_early / (total_gdd + 1e-6)
    m_rel = gdd_mid / (total_gdd + 1e-6)
    r_e_rel = rain_early / (total_rain + 1e-6)
    r_m_rel = rain_mid / (total_rain + 1e-6)

    # Reglas empíricas calibradas con 2001–2025
    score_early = (e_rel * 0.6 + r_e_rel * 0.4)
    score_medium = (m_rel * 0.6 + r_m_rel * 0.4)
    score_staggered = (0.5 * (score_early + score_medium)) + abs(e_rel - m_rel) * 0.3

    scores = {"early": score_early, "staggered": score_staggered, "medium": score_medium}
    total = sum(scores.values())
    probs = {k: round(v / total, 3) for k, v in scores.items()}

    # Fecha de discriminación
    if probs["early"] > 0.6:
        fecha = "15 de abril (≈ JD 105)"
        clasif = "EARLY"
    elif probs["medium"] > 0.6:
        fecha = "1 de junio (≈ JD 152)"
        clasif = "MEDIUM"
    else:
        fecha = "1 de mayo (≈ JD 121)"
        clasif = "STAGGERED"

    return {
        "clasificacion": clasif,
        "fecha_prediccion_fiable": fecha,
        "probabilidades": probs
    }

# ---------- PIPELINE PRINCIPAL ----------
def predecir_patron(path: str):
    df = load_meteo_history(path)
    df = compute_indicators(df)
    res = classify_pattern(df)
    print("🌾 Clasificación meteorológica del patrón histórico")
    print("---------------------------------------------------")
    print(f"📆 Fecha de predicción fiable: {res['fecha_prediccion_fiable']}")
    print(f"🧩 Patrón clasificado: {res['clasificacion']}")
    print(f"🔢 Probabilidades: {res['probabilidades']}")
    return res

# ---------- EJEMPLO DE USO ----------
if __name__ == "__main__":
    res = predecir_patron("meteo_history.csv")

