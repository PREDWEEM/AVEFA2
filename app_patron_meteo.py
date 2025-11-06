# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 AVEFA — Clasificador de patrones meteorológicos (desde imagen o simulación)
# ---------------------------------------------------------------
# - Carga de imagen (pegada o arrastrada)
# - Análisis de patrones por picos y magnitud
# - Clasificación: early / medium / staggered / late
# - Fecha de corte: 1 de junio
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

# === INTENTAR IMPORTAR LIBRERÍAS OPCIONALES ===
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

# === FUNCIÓN ALTERNATIVA PARA DETECTAR PICOS (SI NO HAY SCIPY) ===
def find_local_peaks(y, threshold=0.01):
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i - 1] < y[i] > y[i + 1] and y[i] > threshold:
            peaks.append(i)
    return np.array(peaks)


# === CONFIGURACIÓN GENERAL ===
st.set_page_config(page_title="Clasificador de Patrones AVEFA", layout="centered")
st.title("🌾 Clasificador de Patrones (AVEFA)")
st.markdown("""
Esta herramienta permite **clasificar patrones de emergencia o crecimiento** según los picos
y su magnitud hasta una fecha de corte (por defecto, el **1 de junio**).
""")

# === PARÁMETROS ===
fecha_ref = dt.datetime(2010, 6, 1)
st.sidebar.header("🗓️ Parámetros de análisis")
fecha_str = st.sidebar.date_input("Fecha de corte", value=fecha_ref).strftime("%Y-%m-%d")
fecha_ref = dt.datetime.strptime(fecha_str, "%Y-%m-%d")

# === CARGA DE IMAGEN ===
st.subheader("📸 Cargar imagen o patrón")
uploaded_file = st.file_uploader("Arrastrá o pegá una imagen (png, jpg, jpeg):", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Leer bytes e intentar mostrar
    bytes_data = uploaded_file.read()
    img_array = np.frombuffer(bytes_data, np.uint8)

    if cv2 is not None:
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Imagen cargada", use_column_width=True)
    else:
        st.warning("⚠️ OpenCV no está disponible, la imagen se mostrará sin procesamiento.")
        st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)

    st.markdown("---")

# === SIMULACIÓN DE DATOS (EJEMPLO) ===
st.subheader("🔍 Clasificación de patrones simulados (ejemplo)")

# Simulación de curvas tipo emergencia
fechas = pd.date_range("2010-01-01", "2010-11-01", freq="D")
patrones = {
    "staggered": np.clip(np.sin(np.linspace(0, 10, len(fechas))) * 0.05 + np.random.rand(len(fechas)) * 0.03, 0, 0.12),
    "early": np.clip(np.exp(-0.5 * ((np.linspace(0, 1, len(fechas)) - 0.25) / 0.1) ** 2) * 0.15, 0, 0.15),
    "medium": np.clip(np.exp(-0.5 * ((np.linspace(0, 1, len(fechas)) - 0.45) / 0.1) ** 2) * 0.12, 0, 0.12),
    "late": np.clip(np.exp(-0.5 * ((np.linspace(0, 1, len(fechas)) - 0.75) / 0.1) ** 2) * 0.10, 0, 0.10),
}


# === FUNCIÓN DE CLASIFICACIÓN ===
def clasificar_patron(y, fechas, ref=fecha_ref):
    fechas = pd.to_datetime(fechas)
    mask = fechas <= ref
    y = np.array(y)[mask]
    fechas = np.array(fechas)[mask]

    if len(y) == 0:
        return {"n_picos": 0, "max_y": np.nan, "fecha_pico": None, "tipo": "sin datos"}

    # Detectar picos
    if find_peaks is not None:
        peaks, _ = find_peaks(y, prominence=0.01)
    else:
        peaks = find_local_peaks(y, threshold=0.01)

    n_picos = len(peaks)
    max_y = y[peaks].max() if n_picos > 0 else y.max()

    if n_picos > 0:
        fecha_pico = pd.to_datetime(fechas[peaks[np.argmax(y[peaks])]])
    else:
        fecha_pico = pd.to_datetime(fechas[np.argmax(y)])

    # Clasificación por reglas agronómicas
    if n_picos > 1 and np.std(y) / np.mean(y) > 0.4:
        tipo = "staggered"
    elif fecha_pico < dt.datetime(2010, 4, 1):
        tipo = "early"
    elif fecha_pico <= dt.datetime(2010, 6, 1):
        tipo = "medium"
    else:
        tipo = "late"

    return {
        "n_picos": int(n_picos),
        "max_y": round(float(max_y), 4),
        "fecha_pico": fecha_pico.strftime("%Y-%m-%d"),
        "tipo": tipo
    }


# === APLICACIÓN A LOS PATRONES ===
resultados = {}
for k, v in patrones.items():
    resultados[k] = clasificar_patron(v, fechas)

res_df = pd.DataFrame(resultados).T
st.dataframe(res_df)

# === VISUALIZACIÓN ===
fig, ax = plt.subplots(figsize=(8, 5))
for k, v in patrones.items():
    ax.plot(fechas, v, label=k)
ax.axvline(fecha_ref, color="k", linestyle="--", label=f"Corte: {fecha_ref.strftime('%d-%b')}")
ax.set_title("Clasificación de patrones hasta la fecha de corte")
ax.set_ylabel("Emergencia relativa")
ax.legend()
st.pyplot(fig)

st.success("✅ Clasificación completada correctamente.")
st.markdown("---")
st.markdown("📘 **Tip:** Pegá una imagen de tu gráfico real para almacenarla junto al diagnóstico.")
