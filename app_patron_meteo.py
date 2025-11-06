# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 AVEFA — Clasificador de patrones por picos y magnitud (desde imagen)
# ===============================================================
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import datetime as dt
import pandas as pd

st.title("🌾 Clasificador de patrones (AVEFA) desde imagen")

# === CARGA DE IMAGEN ===
st.write("📸 Pegá o arrastrá la imagen del gráfico aquí:")
uploaded_file = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg"])

fecha_ref = dt.datetime(2010, 6, 1)

if uploaded_file is not None:
    # Convertir la imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Imagen cargada", use_column_width=True)

    st.subheader("🔍 Extracción de curvas y clasificación")

    # === Simulación de extracción (placeholder: en producción se usaría detección de color) ===
    # Supongamos que ya tenemos series extraídas
    fechas = pd.date_range("2010-01-01", "2010-11-01", freq="D")
    patrones = {
        "staggered": np.random.rand(len(fechas)) * 0.1,
        "early": np.random.rand(len(fechas)) * 0.1,
        "medium": np.random.rand(len(fechas)) * 0.1,
    }

    # Suavizado y clasificación
    def clasificar_patron(y, fechas, ref=fecha_ref):
        mask = fechas <= ref
        y = np.array(y)[mask]
        fechas = np.array(fechas)[mask]

        peaks, _ = find_peaks(y, prominence=0.01)
        n_picos = len(peaks)
        max_y = y[peaks].max() if n_picos > 0 else y.max()
        if n_picos > 0:
            fecha_pico = fechas[peaks[np.argmax(y[peaks])]]
        else:
            fecha_pico = fechas[np.argmax(y)]

        if n_picos > 1 and np.std(y) / np.mean(y) > 0.4:
            tipo = "staggered"
        elif fecha_pico < dt.datetime(2010, 4, 1):
            tipo = "early"
        elif fecha_pico <= dt.datetime(2010, 6, 1):
            tipo = "medium"
        else:
            tipo = "late"
        return {
            "n_picos": n_picos,
            "max_y": round(float(max_y), 4),
            "fecha_pico": fecha_pico.date(),
            "tipo": tipo
        }

    # === Aplicar ===
    resultados = {}
    for k, v in patrones.items():
        resultados[k] = clasificar_patron(v, fechas)

    res_df = pd.DataFrame(resultados).T
    st.dataframe(res_df)

    # === Visualización ===
    fig, ax = plt.subplots(figsize=(8,5))
    for k, v in patrones.items():
        ax.plot(fechas, v, label=k)
    ax.axvline(fecha_ref, color='k', linestyle='--', label="1 Jun")
    ax.legend(); ax.set_title("Clasificación hasta 1 de junio")
    st.pyplot(fig)

