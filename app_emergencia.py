# -*- coding: utf-8 -*-
# app_borde2025_merge.py
# Fusiona BORDE2025.csv (histórico) + meteo_history.csv (pronóstico) sin inventar fechas,
# ejecuta la red neuronal y muestra gráficos/tablas dentro del horizonte realmente disponible.
#
# Reglas de fusión por defecto:
#   - Para fechas <= última fecha del histórico (BORDE2025.csv) se usa el histórico.
#   - Para fechas  > última fecha del histórico, se usan filas del pronóstico (meteo_history.csv).
# Si hay fechas duplicadas en ambos, prevalece el histórico en fechas <= último histórico.
#
# Columnas admitidas (flexibles por nombre):
#   - BORDE2025.csv: Fecha/date, TMAX/tmax, TMIN/tmin, Prec/prec, (opcional Julian_days/jd)
#   - meteo_history.csv: date, tmax, tmin, prec[, jd, source, updated_at]
# Salida: EMERREL (barras + MA5) y EMEAC (%) (curva + banda), Tabla con emojis de nivel.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="PREDWEEM · BORDE2025 + Pronóstico (merge)", layout="wide")

PATH_BORDE = Path("BORDE2025.csv")
PATH_PRON  = Path("meteo_history.csv")

# ===================== Utilidades de normalización =====================
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower = {c.lower(): c for c in out.columns}
    def pick(*cands):
        for cc in cands:
            if cc in lower:
                return lower[cc]
        return None
    m = {
        pick("fecha","date"): "Fecha",
        pick("julian_days","jd","julianday","doy"): "Julian_days",
        pick("tmax","tx","t_max"): "TMAX",
        pick("tmin","tn","t_min"): "TMIN",
        pick("prec","ppt","lluvia","prcp"): "Prec",
    }
    m = {k:v for k,v in m.items() if k is not None}
    out = out.rename(columns=m)
    # Parse Fecha
    if "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.normalize()
    # Si no hay Fecha pero sí Julian_days, intentar reconstruir en el año de la primera fila con fecha conocida aparte
    if "Fecha" not in out.columns and "Julian_days" in out.columns:
        year = pd.Timestamp.now().year
        out["Fecha"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(pd.to_numeric(out["Julian_days"], errors="coerce") - 1, unit="D")
    # Tipos numéricos
    for c in ["TMAX","TMIN","Prec","Julian_days"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Sanitizar meteorología
    if "Prec" in out.columns:
        out["Prec"] = out["Prec"].fillna(0).clip(lower=0)
    if {"TMAX","TMIN"}.issubset(out.columns):
        bad = out["TMAX"] < out["TMIN"]
        if bad.any():
            out.loc[bad, ["TMAX","TMIN"]] = out.loc[bad, ["TMIN","TMAX"]].values
    # Ordenar y dejar mínimas columnas
    req = {"Fecha","TMAX","TMIN","Prec"}
    if "Julian_days" not in out.columns and "Fecha" in out.columns:
        out["Julian_days"] = out["Fecha"].dt.dayofyear
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates("Fecha", keep="last").reset_index(drop=True)
    if not req.issubset(out.columns):
        # devolver lo que haya
        return out
    return out[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def load_borde():
    if not PATH_BORDE.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    try:
        if PATH_BORDE.suffix.lower() == ".csv":
            df = pd.read_csv(PATH_BORDE, parse_dates=["Fecha","date"], dayfirst=True, infer_datetime_format=True)
        else:
            df = pd.read_excel(PATH_BORDE)
    except Exception:
        df = pd.read_csv(PATH_BORDE, sep=";", engine="python", dtype=str)
    return _normalize_cols(df)

def load_pron():
    if not PATH_PRON.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    # meteo_history.csv — esquema: date, tmax, tmin, prec (y extras)
    try:
        df = pd.read_csv(PATH_PRON, parse_dates=["date","Fecha"], dayfirst=False)
    except Exception:
        df = pd.read_csv(PATH_PRON)
    # Renombrar a esquema uniforme
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("fecha")
    tmax_col = cols.get("tmax") or cols.get("t_max") or cols.get("tx")
    tmin_col = cols.get("tmin") or cols.get("t_min") or cols.get("tn")
    prec_col = cols.get("prec") or cols.get("ppt") or cols.get("lluvia")
    if date_col is None or tmax_col is None or tmin_col is None or prec_col is None:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    out = pd.DataFrame({
        "Fecha": pd.to_datetime(df[date_col], errors="coerce").dt.normalize(),
        "TMAX": pd.to_numeric(df[tmax_col], errors="coerce"),
        "TMIN": pd.to_numeric(df[tmin_col], errors="coerce"),
        "Prec": pd.to_numeric(df[prec_col], errors="coerce").fillna(0).clip(lower=0),
    })
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates("Fecha", keep="last").reset_index(drop=True)
    out["Julian_days"] = out["Fecha"].dt.dayofyear
    return out[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def merge_hist_forecast(df_hist: pd.DataFrame, df_fc: pd.DataFrame) -> pd.DataFrame:
    if df_hist.empty and df_fc.empty:
        return df_hist
    if df_hist.empty:
        return df_fc.copy()
    if df_fc.empty:
        return df_hist.copy()
    last_hist = df_hist["Fecha"].max()
    # Partes
    past = df_hist[df_hist["Fecha"] <= last_hist]
    future = df_fc[df_fc["Fecha"] > last_hist]
    # Si el forecast trae días anteriores por correcciones, no los usamos por defecto (hist prevalece)
    merged = pd.concat([past, future], ignore_index=True)
    merged = (merged.dropna(subset=["Fecha"])
                    .sort_values("Fecha")
                    .drop_duplicates(subset=["Fecha"], keep="last")
                    .reset_index(drop=True))
    return merged

# ===================== Red neuronal (fallback interna) =====================
class PracticalANNModel:
    def __init__(self):
        self.IW = np.array([
            [-2.924160, -7.896739, -0.977000, 0.554961, 9.510761, 8.739410, 10.592497, 21.705275, -2.532038, 7.847811,
             -3.907758, 13.933289, 3.727601, 3.751941, 0.639185, -0.758034, 1.556183, 10.458917, -1.343551, -14.721089],
            [0.115434, 0.615363, -0.241457, 5.478775, -26.598709, -2.316081, 0.545053, -2.924576, -14.629911, -8.916969,
             3.516110, -6.315180, -0.005914, 10.801424, 4.928928, 1.158809, 4.394316, -23.519282, 2.694073, 3.387557],
            [6.210673, -0.666815, 2.923249, -8.329875, 7.029798, 1.202168, -4.650263, 2.243358, 22.006945, 5.118664,
             1.901176, -6.076520, 0.239450, -6.862627, -7.592373, 1.422826, -2.575074, 5.302610, -6.379549, -14.810670],
            [10.220671, 2.665316, 4.119266, 5.812964, -3.848171, 1.472373, -4.829068, -7.422444, 0.862384, 0.001028,
             0.853059, 2.953289, 1.403689, -3.040909, -6.946802, -1.799923, 0.994357, -5.551789, -0.764891, 5.520776]
        ], dtype=float)
        self.bias_IW = np.array([
            7.229977, -2.428431, 2.973525, 1.956296, -1.155897, 0.907013, 0.231416, 5.258464, 3.284862, 5.474901,
            2.971978, 4.302273, 1.650572, -1.768043, -7.693806, -0.010850, 1.497102, -2.799158, -2.366918, -9.754413
        ], dtype=float)
        self.LW = np.array([
            5.508609, -21.909052, -10.648533, -2.939799, 8.192068, -2.157424, -3.373238, -5.932938, -2.680237,
            -3.399422, 5.870659, -1.720078, 7.134293, 3.227154, -5.039080, -10.872101, -6.569051, -8.455429,
            2.703778, 4.776029
        ], dtype=float)
        self.bias_out = -5.394722
        self.input_min = np.array([1.0, 7.7, -3.5, 0.0], dtype=float)
        self.input_max = np.array([148.0, 38.5, 23.5, 59.9], dtype=float)

    @staticmethod
    def _tansig(x): return np.tanh(x)
    def _normalize_input(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / (self.input_max - self.input_min) - 1

    def predict(self, X):
        Xn = self._normalize_input(X.astype(float))
        z1 = Xn @ self.IW + self.bias_IW
        a1 = self._tansig(z1)
        z2 = a1 @ self.LW + self.bias_out
        y  = self._tansig(z2)             # [-1..1]
        emerrel01 = (y + 1.0) / 2.0       # [0..1]
        acc = np.cumsum(emerrel01)
        emeac01 = acc / 8.05              # denominador de referencia
        emerrel = np.diff(emeac01, prepend=0.0)
        return emerrel, emeac01

# ===================== UI =====================
st.title("BORDE2025 (histórico) + meteo_history.csv (pronóstico) — Fusión sin reindex")

# Cargar archivos
hist = load_borde()
fcst = load_pron()

colA, colB = st.columns(2)
with colA:
    st.markdown("**BORDE2025.csv (histórico)**")
    if hist.empty:
        st.error("No se pudo leer BORDE2025.csv o faltan columnas mínimas.")
    else:
        st.success(f"Histórico: {hist['Fecha'].min().date()} → {hist['Fecha'].max().date()} · {len(hist)} fila(s)")
with colB:
    st.markdown("**meteo_history.csv (pronóstico)**")
    if fcst.empty:
        st.warning("No se pudo leer meteo_history.csv (se usará sólo histórico si existe).")
    else:
        st.info(f"Pronóstico CSV: {fcst['Fecha'].min().date()} → {fcst['Fecha'].max().date()} · {len(fcst)} fila(s)")

merged = merge_hist_forecast(hist, fcst)
if merged.empty:
    st.stop()

st.success(f"Serie fusionada: {merged['Fecha'].min().date()} → {merged['Fecha'].max().date()} · {len(merged)} fila(s)")

# ===================== Modelo =====================
# Vector X = [Julian_days, TMAX, TMIN, Prec]
X = merged[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
model = PracticalANNModel()
emerrel, emeac01 = model.predict(X)

pred = pd.DataFrame({
    "Fecha": merged["Fecha"].to_numpy(),
    "Julian_days": merged["Julian_days"].to_numpy(),
    "EMERREL(0-1)": pd.to_numeric(emerrel, errors="coerce").fillna(0),
    "EMEAC(0-1)": pd.to_numeric(emeac01, errors="coerce").fillna(0),
})
pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
pred["EMEAC(%)"] = (pred["EMEAC(0-1)"] * 100).clip(0, 100)

# Clasificación simple del nivel diario
THR_BAJO_MEDIO = 0.02
THR_MEDIO_ALTO = 0.079
def nivel(v):
    if v < THR_BAJO_MEDIO: return "Bajo"
    elif v <= THR_MEDIO_ALTO: return "Medio"
    else: return "Alto"
pred["Nivel"] = pred["EMERREL(0-1)"].apply(nivel)
iconos = {"Bajo":"🟢 Bajo", "Medio":"🟠 Medio", "Alto":"🔴 Alto"}

# ===================== Figuras =====================
st.subheader("EMERREL diario (barras) + MA5 (línea)")
colores = pred["Nivel"].map({"Bajo":"#2ca02c","Medio":"#ff7f0e","Alto":"#d62728"}).fillna("#808080")
fig1 = go.Figure()
fig1.add_bar(
    x=pred["Fecha"], y=pred["EMERREL(0-1)"],
    marker=dict(color=colores),
    customdata=pred["Nivel"].map(iconos),
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
    name="EMERREL"
)
fig1.add_scatter(x=pred["Fecha"], y=pred["EMERREL_MA5"], mode="lines", name="MA5")
fig1.update_yaxes(range=[0, 0.08])
st.plotly_chart(fig1, use_container_width=True)

st.subheader("EMEAC acumulada (%)")
EMEAC_MIN_DEN = 5.0
EMEAC_MAX_DEN = 15.0
acc = pred["EMERREL(0-1)"].cumsum()
emeac_min = (acc / EMEAC_MIN_DEN * 100).clip(0, 100)
emeac_max = (acc / EMEAC_MAX_DEN * 100).clip(0, 100)
fig2 = go.Figure()
fig2.add_scatter(x=pred["Fecha"], y=emeac_min, mode="lines", line=dict(width=0), name="EMEAC mín")
fig2.add_scatter(x=pred["Fecha"], y=emeac_max, mode="lines", line=dict(width=0), fill="tonexty", name="EMEAC máx")
fig2.add_scatter(x=pred["Fecha"], y=pred["EMEAC(%)"], mode="lines", name="EMEAC (%) (modelo)")
fig2.update_yaxes(range=[0, 100])
st.plotly_chart(fig2, use_container_width=True)

# ===================== Tabla =====================
st.subheader("Resultados diarios (horizonte fusionado)")
tabla = pred[["Fecha","Julian_days","EMERREL(0-1)","EMERREL_MA5","EMEAC(%)","Nivel"]].copy()
tabla["Nivel"] = tabla["Nivel"].map(iconos)
st.dataframe(tabla, use_container_width=True)

st.download_button(
    "Descargar resultados (CSV)",
    tabla.to_csv(index=False).encode("utf-8"),
    "resultados_borde2025_merge.csv",
    "text/csv"
)