# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 APP — Diagnóstico Histórico de Patrones de Emergencia
# Versión 3: gráfico de confianza por año (reemplaza GDD)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Diagnóstico Histórico de Patrones de Emergencia", layout="wide")
st.title("🌾 Diagnóstico Histórico de Patrones de Emergencia (meteo_history multianual)")

TEMP_BASE = 0.0
RAIN_DRY = 1.0

# ---------- CARGA DE DATOS ----------
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

# ---------- CLASIFICADOR ----------
def clasificar_patron(df):
    jd = df["julian_days"].to_numpy()
    gdd = df["gdd"].cumsum().to_numpy()
    rain = df["prec"].cumsum().to_numpy()

    def sum_in_window(v, start, end):
        m = (jd >= start) & (jd <= end)
        return float(np.nansum(v[m])) / max(1, end - start + 1)

    gdd_early, gdd_mid = sum_in_window(gdd, 60, 120), sum_in_window(gdd, 150, 210)
    rain_early, rain_mid = sum_in_window(rain, 60, 120), sum_in_window(rain, 150, 210)
    total_gdd, total_rain = np.nanmax(gdd), np.nanmax(rain)

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

    prob_dom = probs[clasif]
    return clasif, probs, jd_c, prob_dom

# ---------- INTERFAZ ----------
uploaded = st.file_uploader("📁 Cargar archivo meteorológico (multianual)", type=["csv"])
if uploaded is None:
    st.info("Subí tu archivo meteorológico con varias campañas (ej. 2001–2025).")
    st.stop()

df = load_meteo(uploaded)
if df.empty:
    st.error("No se pudieron leer datos válidos.")
    st.stop()

diagnosticos = []
for año, sub in df.groupby("año"):
    clasif, probs, jd_c, prob_dom = clasificar_patron(sub)
    diagnosticos.append({
        "Año": año,
        "Patrón": clasif,
        "Prob_EARLY": probs["EARLY"],
        "Prob_STAGGERED": probs["STAGGERED"],
        "Prob_MEDIUM": probs["MEDIUM"],
        "JD_discriminación": jd_c,
        "Probabilidad_discriminación": round(prob_dom,3)
    })

tabla = pd.DataFrame(diagnosticos).sort_values("Año")
st.subheader("📊 Clasificación histórica por año")
st.dataframe(tabla, use_container_width=True)

# ---------- GRAFICO DE CONFIANZA ----------
st.subheader("📈 Confianza del patrón clasificado por año")
colors = {"EARLY": "#00A651", "STAGGERED": "#FFC107", "MEDIUM": "#1976D2"}

fig_conf = go.Figure()
for _, row in tabla.iterrows():
    fig_conf.add_trace(go.Bar(
        x=[row["Año"]],
        y=[row["Probabilidad_discriminación"]*100],
        name=row["Patrón"],
        marker_color=colors[row["Patrón"]],
        text=f"{row['Patrón']} ({row['Probabilidad_discriminación']*100:.1f}%)",
        textposition="auto"
    ))

fig_conf.update_layout(
    barmode="group",
    xaxis_title="Año",
    yaxis_title="Confianza del patrón (%)",
    yaxis=dict(range=[0,100]),
    hovermode="x unified",
    legend_title="Patrón clasificado",
    height=500
)
st.plotly_chart(fig_conf, use_container_width=True)

# ---------- INTERPRETACIÓN ----------
st.markdown("---")
st.subheader("🧠 Interpretación agronómica")
st.write("""
**Días de discriminación y confiabilidad:**
- JD **105 (15 abril)** → EARLY → confianza ≥ **90%**
- JD **121 (1 mayo)** → STAGGERED → confianza ≥ **85–90%**
- JD **152 (1 junio)** → MEDIUM → confianza ≥ **90%**

**Lectura del gráfico:**
- Barras altas (≥90%) indican pronósticos **muy certeros**.  
- Barras entre 75–85% muestran **patrones mixtos o años transicionales**.
""")

