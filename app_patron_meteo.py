# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 APP — Sondeo del mejor día de discriminación (JD óptimo)
# Usa meteo_history.csv para determinar el día con mayor poder predictivo
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Sondeo Día de Discriminación", layout="wide")
st.title("🌾 Sondeo del mejor día de discriminación según meteo_history.csv")

# ------------------- CONFIG -------------------
TEMP_BASE = 0.0
RAIN_DRY = 1.0

# ------------------- CARGA -------------------
@st.cache_data(ttl=600)
def load_meteo(path):
    df = pd.read_csv(path, sep=";", decimal=",", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
        df["año"] = df["fecha"].dt.year
        df["julian_days"] = df["fecha"].dt.dayofyear
    df["tmax"] = pd.to_numeric(df.get("tmax", np.nan), errors="coerce")
    df["tmin"] = pd.to_numeric(df.get("tmin", np.nan), errors="coerce")
    df["prec"] = pd.to_numeric(df.get("prec", np.nan), errors="coerce").clip(lower=0)
    df["tmed"] = (df["tmax"] + df["tmin"]) / 2
    df["gdd"] = np.maximum(df["tmed"] - TEMP_BASE, 0)
    return df.dropna(subset=["tmed"])

uploaded = st.file_uploader("📁 Cargar meteo_history.csv", type=["csv"])
if uploaded is None:
    st.info("Subí tu archivo meteo_history.csv para analizar.")
    st.stop()

df = load_meteo(uploaded)

# ------------------- FUNCIÓN DE ANÁLISIS -------------------
def evaluar_discriminacion(df, jd_test):
    """Evalúa qué tan bien el JD separa los patrones históricos"""
    resultados = []
    for año, sub in df.groupby("año"):
        gdd_acum = sub.loc[sub["julian_days"] <= jd_test, "gdd"].sum()
        lluvia_acum = sub.loc[sub["julian_days"] <= jd_test, "prec"].sum()
        ratio = (gdd_acum / (lluvia_acum + 1e-6))
        resultados.append(ratio)

    arr = np.array(resultados)
    media = np.nanmean(arr)
    dispersion = np.nanstd(arr)
    confianza = 1 - (dispersion / (media + 1e-6))  # mayor homogeneidad = mayor confianza
    return confianza

# ------------------- BÚSQUEDA DEL MEJOR JD -------------------
jd_range = range(60, 220)
probs = []
for jd in jd_range:
    c = evaluar_discriminacion(df, jd)
    probs.append(c)

df_eval = pd.DataFrame({"JD": list(jd_range), "Confianza": probs})
best_idx = df_eval["Confianza"].idxmax()
jd_optimo = int(df_eval.loc[best_idx, "JD"])
conf_max = float(df_eval.loc[best_idx, "Confianza"])

st.success(f"📅 Día óptimo de discriminación: **JD {jd_optimo}** con confianza máxima de **{conf_max:.2f}**")

# ------------------- GRÁFICO DE CONFIANZA -------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_eval["JD"], y=df_eval["Confianza"],
    mode="lines+markers",
    line=dict(width=2, color="#007ACC"),
    hovertemplate="JD %{x}<br>Confianza %{y:.3f}<extra></extra>",
))
fig.add_vline(
    x=jd_optimo, line_color="red", line_dash="dash",
    annotation_text=f"JD óptimo = {jd_optimo} ({conf_max:.2f})", annotation_position="top"
)
fig.update_layout(
    title="Curva de discriminación por JD",
    xaxis_title="Día Juliano (JD)",
    yaxis_title="Confianza del patrón (0–1)",
    yaxis=dict(range=[0, 1]),
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# ------------------- INTERPRETACIÓN -------------------
st.markdown("---")
st.subheader("🧠 Interpretación")
st.write(f"""
El modelo sondeó todos los días julianos entre 60 y 220 (marzo a agosto) para evaluar
cuál separa mejor los patrones históricos.

**Resultado:**
- Día óptimo de discriminación → **JD {jd_optimo}**
- Confianza máxima → **{conf_max:.2f}**
- Esto indica que alrededor del **día {jd_optimo}**, las condiciones térmico-hídricas
  son más homogéneas entre años, y por tanto más estables para usar como punto de corte
  en la predicción de emergencia.
""")


