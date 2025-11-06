# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 APP — Predicción del Patrón Histórico de Emergencia (AVEFA / PREDWEEM)
# Versión 3: Diagnóstico definitivo anclado al 1 de junio (JD ≈ 152)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------- CONFIGURACIÓN UI ----------
st.set_page_config(page_title="Predicción del Patrón Histórico de Emergencia", layout="wide")
st.markdown("<style>#MainMenu, header, footer {visibility: hidden;}</style>", unsafe_allow_html=True)
st.title("🌾 Predicción del Patrón Histórico de Emergencia (meteo_history.csv)")

# ---------- PARÁMETROS ----------
TEMP_BASE = 0.0
RAIN_DRY = 1.0
MONTH_START = 9
YEAR_REF = 2025
JD_DIAG = 152  # 1 de junio

# ---------- CARGA DE DATOS ----------
@st.cache_data(ttl=600)
def load_meteo_history(path: str = "meteo_history.csv") -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=";", decimal=",", engine="python")
    except Exception:
        df = pd.read_csv(path, engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "julian_days" not in df.columns:
        df["julian_days"] = np.arange(1, len(df) + 1)
    df["tmax"] = pd.to_numeric(df.get("tmax", df.get("tx", np.nan)), errors="coerce")
    df["tmin"] = pd.to_numeric(df.get("tmin", df.get("tn", np.nan)), errors="coerce")
    df["prec"] = pd.to_numeric(df.get("prec", df.get("ppt", np.nan)), errors="coerce").clip(lower=0)
    df["tmed"] = (df["tmax"] + df["tmin"]) / 2
    df["gdd"] = np.maximum(df["tmed"] - TEMP_BASE, 0)
    df["rainy"] = (df["prec"] >= RAIN_DRY).astype(int)
    return df.dropna(subset=["tmed"])

def compute_indicators(df):
    df["gdd_cum"] = df["gdd"].cumsum()
    df["rain_cum"] = df["prec"].cumsum()
    df["humid30"] = df["rainy"].rolling(30, min_periods=1).sum()
    df["ih_thermal"] = df["gdd_cum"] * (1 + df["rain_cum"] / 100)
    return df

def classify_pattern(df):
    jd = df["julian_days"].to_numpy()
    gdd = df["gdd_cum"].to_numpy()
    rain = df["rain_cum"].to_numpy()

    def sum_in_window(v, start, end):
        m = (jd >= start) & (jd <= end)
        return float(np.nansum(v[m])) / max(1, end - start + 1)

    gdd_early, gdd_mid = sum_in_window(gdd, 60, 120), sum_in_window(gdd, 150, 210)
    rain_early, rain_mid = sum_in_window(rain, 60, 120), sum_in_window(rain, 150, 210)

    total_gdd, total_rain = np.nanmax(gdd), np.nanmax(rain)
    e_rel, m_rel = gdd_early / (total_gdd+1e-6), gdd_mid / (total_gdd+1e-6)
    r_e_rel, r_m_rel = rain_early / (total_rain+1e-6), rain_mid / (total_rain+1e-6)

    score_early = (e_rel * 0.6 + r_e_rel * 0.4)
    score_medium = (m_rel * 0.6 + r_m_rel * 0.4)
    score_staggered = (0.5 * (score_early + score_medium)) + abs(e_rel - m_rel) * 0.3

    scores = {"early": score_early, "staggered": score_staggered, "medium": score_medium}
    total = sum(scores.values())
    probs = {k: round(v / total, 3) for k, v in scores.items()}

    # Clasificación con diagnóstico al 1 de junio
    clasif = max(probs, key=probs.get).upper()
    fecha_diag = "1 de junio (JD 152)"
    return clasif, fecha_diag, probs

uploaded = st.file_uploader("📁 Cargar meteo_history.csv", type=["csv"])
df = load_meteo_history(uploaded) if uploaded else load_meteo_history("meteo_history.csv")
if df.empty: st.error("No se pudieron cargar los datos."); st.stop()
df = compute_indicators(df)

# ---------- CLASIFICACIÓN ----------
clasif, fecha, probs = classify_pattern(df)

# ---------- VISUALIZACIÓN ----------
col1, col2 = st.columns([1.2, 1])
with col1:
    st.metric("🧩 Patrón clasificado", clasif)
    st.write(f"📆 Fecha de diagnóstico definitivo: **{fecha}**")
    st.write("🔢 Probabilidades estimadas:")
    st.json(probs)
with col2:
    val = probs[clasif.lower()] * 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': f"Probabilidad {clasif}", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00A651" if clasif=="EARLY" else "#E5C700" if clasif=="STAGGERED" else "#1976D2"},
            'steps': [
                {'range': [0, 50], 'color': '#EEEEEE'},
                {'range': [50, 100], 'color': '#B3E5FC'}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------- BARRA TEMPORAL ----------
st.subheader("📆 Línea temporal (diagnóstico fijo 1 de junio)")
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=df["julian_days"], y=df["gdd_cum"], mode="lines", name="GDD acumulados", line=dict(color="red", width=2)))
fig_line.add_trace(go.Scatter(x=df["julian_days"], y=df["rain_cum"], mode="lines", name="Lluvia acumulada (mm)", line=dict(color="blue", width=2)))
fig_line.add_vline(x=JD_DIAG, line_width=3, line_dash="dash", line_color="green", annotation_text="Diagnóstico (1-Jun)", annotation_position="top")
fig_line.update_layout(xaxis_title="Día Juliano (JD)", yaxis_title="Valor acumulado", hovermode="x unified", height=500, legend_title="Variables")
st.plotly_chart(fig_line, use_container_width=True)

# ---------- INTERPRETACIÓN AGRONÓMICA ----------
st.markdown("---")
st.subheader("🧠 Interpretación agronómica (Diagnóstico 1 de junio)")
if clasif == "EARLY":
    st.success("🌱 **Patrón EARLY** — Emergencia concentrada en marzo–abril. Control presiembra y preemergente crucial.")
elif clasif == "STAGGERED":
    st.warning("🌾 **Patrón STAGGERED** — Emergencia escalonada (2+ cohortes). Residual prolongado + monitoreo hasta julio.")
else:
    st.info("❄️ **Patrón MEDIUM** — Emergencia invernal tardía. Requiere control residual largo o postemergente invernal.")

