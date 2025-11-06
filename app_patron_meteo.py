# -*- coding: utf-8 -*-
# ===============================================================
# 🌾 AVEFA — Clasificador híbrido (color o monocromo)
# ---------------------------------------------------------------
# - Si hay color → detecta curvas azul/naranja/gris (staggered/early/medium)
# - Si es monocromo → detecta puntos o trazos negros y genera 1 sola serie
# - Calcula picos, tipo, y % de confianza
# ===============================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.signal import find_peaks
except ImportError:
    find_peaks = None

def find_local_peaks(y, threshold=0.01):
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i - 1] < y[i] > y[i + 1] and y[i] > threshold:
            peaks.append(i)
    return np.array(peaks)

st.set_page_config(page_title="Clasificador AVEFA híbrido", layout="wide")
st.title("🌾 Clasificador de Patrones AVEFA (Color + Monocromo)")

st.sidebar.header("🛠️ Parámetros de análisis")
default_year = 2011
fecha_corte = st.sidebar.date_input("Fecha de corte", value=dt.date(default_year, 6, 1))
fecha_corte = dt.datetime.combine(fecha_corte, dt.time())

ini = dt.datetime(default_year, 1, 1)
fin = dt.datetime(default_year, 11, 1)
dates = pd.date_range(ini, fin, 365)

uploaded = st.file_uploader("📸 Cargá o pegá tu gráfico (PNG/JPG):", type=["png","jpg","jpeg"])
if uploaded is None:
    st.info("Cargá una imagen para continuar.")
    st.stop()

if cv2 is None:
    st.error("Falta `opencv-python-headless`. Instálalo para análisis de imagen.")
    st.stop()

# === Leer imagen ===
data = np.frombuffer(uploaded.read(), np.uint8)
img = cv2.imdecode(data, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# === Detección de saturación media ===
hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
mean_sat = hsv[..., 1].mean()

if mean_sat < 30:
    modo = "monocromo"
else:
    modo = "color"
st.caption(f"🔍 Detección automática: **{modo.upper()}** (Saturación media={mean_sat:.1f})")

# === Procesamiento según modo ===
kernel = np.ones((3,3), np.uint8)

def mask_to_series(mask):
    H, W = mask.shape
    ys = np.full(W, np.nan)
    for x in range(W):
        idx = np.where(mask[:, x] > 0)[0]
        if idx.size > 0:
            ys[x] = idx.mean()
    if np.sum(~np.isnan(ys)) < 5:
        return None
    x = np.arange(W)
    ys = np.interp(x, x[~np.isnan(ys)], ys[~np.isnan(ys)])
    ys = 1 - (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)
    return ys

series = {}

if modo == "color":
    # Detectar tres curvas por HSV
    mask_blue = cv2.inRange(hsv, (90, 60, 60), (130, 255, 255))
    mask_orange = cv2.inRange(hsv, (5, 100, 60), (25, 255, 255))
    mask_gray = cv2.inRange(hsv, (0, 0, 50), (180, 50, 200))
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
    mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
    series = {
        "staggered": mask_to_series(mask_blue),
        "early": mask_to_series(mask_orange),
        "medium": mask_to_series(mask_gray)
    }
else:
    # === modo monocromático ===
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = cv2.inRange(gray, 0, 180)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    series = {"principal": mask_to_series(mask)}

# === Clasificación ===
def clasificar(y, fechas, ref):
    if y is None:
        return None
    mask = fechas <= ref
    y = np.array(y)[mask]
    fechas = np.array(fechas)[mask]
    if len(y) < 5:
        return None

    # suavizado
    y_smooth = np.convolve(y, np.ones(5)/5, mode="same")

    # picos
    if find_peaks is not None:
        peaks, props = find_peaks(y_smooth, prominence=0.02)
        prom = np.mean(props.get("prominences", [0]))
    else:
        peaks = find_local_peaks(y_smooth, 0.02)
        prom = 0.02
    n_picos = len(peaks)
    max_y = float(y_smooth.max())
    fecha_pico = fechas[peaks[np.argmax(y_smooth[peaks])]] if n_picos else fechas[np.argmax(y_smooth)]

    # tipo
    if n_picos > 1 and np.std(y_smooth)/max(np.mean(y_smooth), 1e-6) > 0.4:
        tipo = "staggered"
    elif fecha_pico < dt.datetime(ref.year, 4, 1):
        tipo = "early"
    elif fecha_pico <= dt.datetime(ref.year, 6, 1):
        tipo = "medium"
    else:
        tipo = "late"

    # confianza
    cv = np.std(y_smooth)/max(np.mean(y_smooth), 1e-9)
    confianza = 65 + prom*120 + cv*10
    confianza = float(np.clip(confianza, 0, 99.9))

    return {
        "n_picos": n_picos,
        "max_y": round(max_y, 4),
        "fecha_pico": fecha_pico.strftime("%Y-%m-%d"),
        "tipo": tipo,
        "confianza": round(confianza, 1),
        "y_full": y_smooth,
        "fechas": fechas
    }

resultados = {k: clasificar(v, dates, fecha_corte) for k,v in series.items()}

# === Mostrar resultados ===
rows = []
for k, r in resultados.items():
    if r is None:
        rows.append({"patrón": k, "estado": "sin detección"})
    else:
        rows.append({
            "patrón": k,
            "n_picos": r["n_picos"],
            "max_y": r["max_y"],
            "fecha_pico": r["fecha_pico"],
            "tipo": r["tipo"],
            "confianza_%": r["confianza"]
        })
df = pd.DataFrame(rows)
st.subheader("📊 Resultado de clasificación")
st.dataframe(df)

# === Gráfico reconstruido ===
st.subheader("📈 Reconstrucción")
fig, ax = plt.subplots(figsize=(8,4))
for k, r in resultados.items():
    if r is None: continue
    ax.plot(r["fechas"], r["y_full"], label=f"{k} · {r['tipo']} ({r['confianza']}%)")
ax.axvline(fecha_corte, color="k", linestyle="--")
ax.legend(); ax.set_ylabel("Intensidad relativa (normalizada)")
ax.set_title("Clasificación automática")
st.pyplot(fig)

st.success("✅ Imagen procesada correctamente.")
