# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 APP — Diagnóstico Histórico de Patrones de Emergencia
# Basado en datos meteorológicos anuales (meteo_history.csv)
# Clasifica cada año como EARLY / STAGGERED / MEDIUM
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------- CONFIGURACIÓN ----------------
st.set_page_config(page_title="Diagnóstico Histórico de Patrones de Emergencia", layout="wide")
st.title("🌾 Diagnóstico Histórico de Patrones de Emergencia")

TEMP_BASE = 0.0
RAIN_DRY = 1.0

# ---------------- CARGA DE DATOS ----------------
@st.cache_data(ttl=600)
def load_meteo(path):
    df = pd.read_csv(path, sep=";", decimal=",", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
        df["año"] = df["fecha"].dt.year
        df["julian_days"] = df["fecha"].dt.dayofyear
    elif "julian_days" in df.columns:
        df["año"] = 2025
    else:
        raise ValueError("El archivo debe contener 'Fecha' o 'Julian_days'.")
    df["tmax"] = pd.to_numeric(df.get("tmax", df.get("tx", np.nan)), errors="coerce")
    df["tmin"] = pd.to_numeric(df.get("tmin", df.get("tn", np.nan)), errors="coerce")
    df["prec"] = pd.to_numeric(df.get("prec", df.get("ppt", np.nan)), errors="coerce").clip(lower=0)
    df["tmed"] = (df["tmax"] + df["tmin"]) / 2
    df["gdd"] = np.maximum(df["tmed"] - TEMP_BASE, 0)
    df["rainy"] = (df["prec"] >= RAIN_DRY).astype(int)
    return df.dropna(subset=["tmed"])

# ---------------- CÁLCULOS POR AÑO ----------------
def indicadores_anuales(df):
    resumen = []
    for año, sub in df.groupby("año"):
        gdd = sub["gdd"].cumsum().iloc[-1]
        lluvia = sub["prec"].sum()
        dias_lluvia = sub["rainy"].sum()
        resumen.append(dict(Año=año, GDD=gdd, Lluvia_mm=lluvia, Días_lluviosos=dias_lluvia))
    return pd.DataFrame(resumen)

def clasificar_patron(df):
    jd = df["julian_days"].to_numpy()
    gdd = df["gdd"].cumsum().to_numpy()
    rain = df["prec"].cumsum().to_numpy()
    def sum_in_window(v, start, end):
        m = (jd >= start) & (jd <= end)
        return float(np.nansum(v[m])) / max(1, end - start + 1)
    gdd_early = sum_in_window(gdd, 60, 120)
    gdd_mid = sum_in_window(gdd, 150, 210)
    rain_early = sum_in_window(rain, 60, 120)
    rain_mid = sum_in_window(rain, 150, 210)
    total_gdd = np.nanmax(gdd)
    total_rain = np.nanmax(rain)
    e_rel, m_rel = gdd_early / (total_gdd+1e-6), gdd_mid / (total_gdd+1e-6)
    r_e_rel, r_m_rel = rain_early / (total_rain+1e-6), rain_mid / (total_rain+1e-6)
    s_early = e_rel*0.6 + r_e_rel*0.4
    s_med = m_rel*0.6 + r_m_rel*0.4
    s_stag = (0.5*(s_early+s_med)) + abs(e_rel - m_rel)*0.3
    total = s_early + s_med + s_stag
    probs = {k: round(v/total,3) for k,v in zip(["EARLY","STAGGERED","MEDIUM"], [s_early,s_stag,s_med])}
    if probs["EARLY"]>0.6: clasif, jd_c = "EARLY", 105
    elif probs["MEDIUM"]>0.6: clasif, jd_c = "MEDIUM", 152
    else: clasif, jd_c = "STAGGERED", 121
    return clasif, probs, jd_c

# ---------------- INTERFAZ ----------------
uploaded = st.file_uploader("📁 Cargar archivo meteorológico (meteo_history.csv o multianual)", type=["csv"])
if uploaded is None:
    st.info("Subí tu archivo meteorológico para clasificar los patrones históricos.")
    st.stop()

df = load_meteo(uploaded)
if df.empty:
    st.error("No se pudieron leer datos válidos.")
    st.stop()

resumen = indicadores_anuales(df)
diagnosticos = []

for año, sub in df.groupby("año"):
    clasif, probs, jd_c = clasificar_patron(sub)
    diagnosticos.append({
        "Año": año,
        "Patrón": clasif,
        "Prob_EARLY": probs["EARLY"],
        "Prob_STAGGERED": probs["STAGGERED"],
        "Prob_MEDIUM": probs["MEDIUM"],
        "JD_discriminación": jd_c
    })

tabla = pd.DataFrame(diagnosticos)
tabla = tabla.merge(resumen, on="Año", how="left")

st.subheader("📊 Clasificación de patrones históricos")
st.dataframe(tabla, use_container_width=True)

# ---------------- GRAFICO COMPARATIVO ----------------
st.subheader("📈 Evolución térmica e hídrica por año")
fig = go.Figure()
for año, sub in df.groupby("año"):
    fig.add_trace(go.Scatter(x=sub["julian_days"], y=sub["gdd"].cumsum(),
                             mode="lines", name=f"GDD {año}"))
fig.update_layout(
    xaxis_title="Día Juliano", yaxis_title="GDD acumulados",
    height=500, hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- INTERPRETACIÓN ----------------
st.markdown("---")
st.subheader("🧠 Interpretación agronómica global")
st.write("""
- **EARLY:** Emergencia concentrada en marzo–abril. Requiere control presiembra y preemergente eficaz.  
- **STAGGERED:** Emergencia escalonada en varias cohortes. Residual prolongado + monitoreo hasta julio.  
- **MEDIUM:** Emergencia invernal tardía (junio–agosto). Precisa control residual largo o postemergente invernal.
""")
