# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 APP — Predicción del Patrón Histórico de Emergencia (AVEFA / PREDWEEM)
# Versión 4: Diagnóstico fijo 1 de junio + evolución diaria de probabilidades
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

# ---------- CLASIFICADOR (vectorizable para todos los días) ----------
def compute_probs(df):
    jd = df["julian_days"].to_numpy()
    gdd = df["gdd_cum"].to_numpy()
    rain = df["rain_cum"].to_numpy()
    total_gdd, total_rain = np.nanmax(gdd), np.nanmax(rain)
    probs = {"early": [], "staggered": [], "medium": []}

    for i in range(len(jd)):
        sub = jd <= jd[i]
        gdd_sub, rain_sub = gdd[sub], rain[sub]

        def sum_in_window(v, start, end):
            m = (jd[sub] >= start) & (jd[sub] <= end)
            return float(np.nansum(v[m])) / max(1, end - start + 1)

        gdd_early, gdd_mid = sum_in_window(gdd_sub, 60, 120), sum_in_window(gdd_sub, 150, 210)
        rain_early, rain_mid = sum_in_window(rain_sub, 60, 120), sum_in_window(rain_sub, 150, 210)

        e_rel = gdd_early / (total_gdd + 1e-6)
        m_rel = gdd_mid / (total_gdd + 1e-6)
        r_e_rel = rain_early / (total_rain + 1e-6)
        r_m_rel = rain_mid / (total_rain + 1e-6)

        score_early = (e_rel * 0.6 + r_e_rel * 0.4)
        score_medium = (m_rel * 0.6 + r_m_rel * 0.4)
        score_staggered = (0.5 * (score_early + score_medium)) + abs(e_rel - m_rel) * 0.3
        total = score_early + score_staggered + score_medium
        probs["early"].append(score_early / total)
        probs["staggered"].append(score_staggered / total)
        probs["medium"].append(score_medium / total)

    for k in probs: probs[k] = np.array(probs[k])
    return probs

# ---------- EJECUCIÓN ----------
uploaded = st.file_uploader("📁 Cargar meteo_history.csv", type=["csv"])
df = load_meteo_history(uploaded) if uploaded else load_meteo_history("meteo_history.csv")
if df.empty: st.error("No se pudieron cargar los datos."); st.stop()
df = compute_indicators(df)
probs = compute_probs(df)

# ---------- DIAGNÓSTICO FINAL (JD 152) ----------
df_diag = df[df["julian_days"] <= JD_DIAG]
final_idx = (np.abs(df["julian_days"] - JD_DIAG)).argmin()
final_probs = {k: round(float(probs[k][final_idx]), 3) for k in probs}
clasif = max(final_probs, key=final_probs.get).upper()

# ---------- VISUALIZACIÓN PRINCIPAL ----------
col1, col2 = st.columns([1.2, 1])
with col1:
    st.metric("🧩 Patrón clasificado (diagnóstico 1 de junio)", clasif)
    st.json(final_probs)
with col2:
    val = final_probs[clasif.lower()] * 100
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        title={'text': f"Probabilidad {clasif}", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00A651" if clasif=="EARLY" else "#E5C700" if clasif=="STAGGERED" else "#1976D2"},
            'steps': [{'range': [0, 50], 'color': '#EEEEEE'}, {'range': [50, 100], 'color': '#B3E5FC'}]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------- EVOLUCIÓN TEMPORAL DE PROBABILIDADES ----------
st.subheader("📈 Evolución diaria de probabilidad de patrón (hasta 1 de junio)")
fig_prob = go.Figure()
for k, col, name in [
    ("early", "#00A651", "EARLY"),
    ("staggered", "#E5C700", "STAGGERED"),
    ("medium", "#1976D2", "MEDIUM"),
]:
    fig_prob.add_trace(go.Scatter(
        x=df["julian_days"], y=probs[k]*100, mode="lines",
        name=f"{name}", line=dict(width=3, color=col)
    ))
fig_prob.add_vline(x=JD_DIAG, line_width=3, line_dash="dash", line_color="green",
                   annotation_text="1 de junio (diagnóstico definitivo)", annotation_position="top")
fig_prob.update_layout(
    yaxis_title="Probabilidad (%)", xaxis_title="Día Juliano (JD)",
    hovermode="x unified", height=500, legend_title="Patrón"
)
st.plotly_chart(fig_prob, use_container_width=True)

# ---------- CURVAS ACUMULADAS DE CLIMA ----------
st.subheader("🌡️ Acumulados térmicos e hídricos")
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=df["julian_days"], y=df["gdd_cum"], mode="lines", name="GDD acumulados", line=dict(color="red", width=2)))
fig_line.add_trace(go.Scatter(x=df["julian_days"], y=df["rain_cum"], mode="lines", name="Lluvia acumulada (mm)", line=dict(color="blue", width=2)))
fig_line.add_vline(x=JD_DIAG, line_width=3, line_dash="dash", line_color="green", annotation_text="1 de junio", annotation_position="top")
fig_line.update_layout(xaxis_title="Día Juliano (JD)", yaxis_title="Valor acumulado", hovermode="x unified", height=500)
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

