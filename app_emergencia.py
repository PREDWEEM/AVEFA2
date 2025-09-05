# -*- coding: utf-8 -*-
# app_emergencia.py
# Fusión BORDE2025 (histórico) + meteo_history (pronóstico), sin reindexar ni inventar fechas.
# Usa solo filas existentes. Red neuronal con entradas [Julian_days, TMAX, TMIN, Prec].
# Incluye: limpieza opcional, base de temporada (JD=1), recorte JD<=148, QA y depuración.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Emergencia · BORDE2025 + meteo_history", layout="wide")

# ===================== Rutas =====================
BORDE_CLEAN = Path("BORDE2025_clean.csv")
BORDE_RAW   = Path("BORDE2025.csv")
PATH_BORDE  = BORDE_CLEAN if BORDE_CLEAN.exists() else BORDE_RAW
PATH_PRON   = Path("meteo_history.csv")

# ===================== Helpers =====================
def _to_num(series: pd.Series) -> pd.Series:
    """Convierte a float aceptando coma decimal, recorta miles (puntos) y limpia caracteres extraños."""
    s = series.astype(str).str.strip()
    s = s.str.replace("​", "", regex=False)  # zero-width
    s = s.str.replace(".", "", regex=False)       # miles
    s = s.str.replace(",", ".", regex=False)      # coma->punto
    s = s.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)  # fuera de dígitos/signo/punto
    return pd.to_numeric(s, errors="coerce")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza a columnas ['Fecha','Julian_days','TMAX','TMIN','Prec'] lo que venga de BORDE/otros."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff","") for c in out.columns]
    lower = {c.lower(): c for c in out.columns}

    def pick(*cands):
        for cc in cands:
            if cc in lower:
                return lower[cc]
        return None

    # Fecha
    fecha_col = pick("fecha", "date", "dia", "día")
    if fecha_col is None and set(out.columns[:4]) == set(["FECHA","TMIN","TMAX","PREC"]):
        fecha_col = "FECHA"  # header-less fallback ya nombrado
    if fecha_col is not None:
        out["Fecha"] = pd.to_datetime(out[fecha_col].astype(str).str.strip(),
                                      errors="coerce", dayfirst=True).dt.normalize()
    else:
        # Intento construir desde año/mes/día
        ycol = pick("año","anio","ano","year"); mcol = pick("mes","month"); dcol = pick("dia","día","day")
        if ycol and mcol and dcol:
            y = pd.to_numeric(out[ycol], errors="coerce").astype("Int64")
            m = pd.to_numeric(out[mcol], errors="coerce").astype("Int64")
            d = pd.to_numeric(out[dcol], errors="coerce").astype("Int64")
            out["Fecha"] = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce").dt.normalize()
        else:
            out["Fecha"] = pd.NaT

    # Meteorología
    tmax_col = pick("tmax","tx","t_max")
    tmin_col = pick("tmin","tn","t_min")
    prec_col = pick("prec","ppt","lluvia","prcp","pp")

    if tmax_col in out.columns: out["TMAX"] = _to_num(out[tmax_col])
    if tmin_col in out.columns: out["TMIN"] = _to_num(out[tmin_col])
    if prec_col in out.columns: out["Prec"] = _to_num(out[prec_col])

    # Defaults y saneo
    if "Prec" in out.columns:
        out["Prec"] = out["Prec"].fillna(0).clip(lower=0)
    if {"TMAX","TMIN"}.issubset(out.columns):
        bad = out["TMAX"] < out["TMIN"]
        if bad.any():
            out.loc[bad, ["TMAX","TMIN"]] = out.loc[bad, ["TMIN","TMAX"]].values

    out = (out.dropna(subset=["Fecha"])
              .sort_values("Fecha")
              .drop_duplicates("Fecha", keep="last")
              .reset_index(drop=True))
    if "Julian_days" not in out.columns:
        out["Julian_days"] = out["Fecha"].dt.dayofyear

    for c in ["TMAX","TMIN","Prec"]:
        if c not in out.columns:
            out[c] = np.nan

    return out[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

# ===================== Carga BORDE =====================
def load_borde() -> pd.DataFrame:
    path_choice = "BORDE2025_clean.csv" if BORDE_CLEAN.exists() else "BORDE2025.csv"
    st.caption(f"Histórico: usando **{path_choice}**" if Path(path_choice).exists() else "Histórico: **no encontrado**")
    p = Path(path_choice)
    if not p.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

    df = None
    # 1) Auto-separador con coma decimal
    try:
        df = pd.read_csv(p, sep=None, engine="python", decimal=",")
    except Exception:
        df = None
    # 2) Intentos con separadores comunes
    if df is None or df.shape[1] < 2:
        for sep in [";","\t",",","|"]:
            try:
                df = pd.read_csv(p, sep=sep, engine="python", decimal=",")
                if df.shape[1] >= 2:
                    break
            except Exception:
                df = None
    # 3) Fallback header-less típico (FECHA;TMIN;TMAX;PREC)
    if df is None or df.columns.tolist()[:4] == (list(df.iloc[0])[:4] if df is not None and len(df)>0 else []):
        try:
            df = pd.read_csv(p, sep=";", engine="python", header=None, names=["FECHA","TMIN","TMAX","PREC"], dtype=str)
        except Exception:
            return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    return _normalize_cols(df)

# ===================== Carga PRONÓSTICO =====================
def load_pron() -> pd.DataFrame:
    p = PATH_PRON
    st.caption(f"Pronóstico: usando **{p.name}**" if p.exists() else "Pronóstico: **no encontrado**")
    if not p.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    # Intento flexible
    try:
        df = pd.read_csv(p)
    except Exception:
        df = pd.read_csv(p, engine="python")
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("fecha")
    tmax_col = cols.get("tmax") or cols.get("t_max") or cols.get("tx")
    tmin_col = cols.get("tmin") or cols.get("t_min") or cols.get("tn")
    prec_col = cols.get("prec") or cols.get("ppt") or cols.get("lluvia") or cols.get("prcp")
    out = pd.DataFrame({
        "Fecha": pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT,
        "TMAX": _to_num(df[tmax_col]) if tmax_col else np.nan,
        "TMIN": _to_num(df[tmin_col]) if tmin_col else np.nan,
        "Prec": _to_num(df[prec_col]).fillna(0).clip(lower=0) if prec_col else 0,
    })
    out = (out.dropna(subset=["Fecha"])
              .sort_values("Fecha")
              .drop_duplicates("Fecha", keep="last")
              .reset_index(drop=True))
    out["Julian_days"] = out["Fecha"].dt.dayofyear
    return out[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

# ===================== Fusión =====================
def merge_hist_forecast(df_hist: pd.DataFrame, df_fc: pd.DataFrame) -> pd.DataFrame:
    if df_hist.empty and df_fc.empty:
        return df_hist
    if df_hist.empty:
        return df_fc.copy()
    if df_fc.empty:
        return df_hist.copy()
    last_hist = df_hist["Fecha"].max()
    past = df_hist[df_hist["Fecha"] <= last_hist]
    future = df_fc[df_fc["Fecha"] > last_hist]
    merged = pd.concat([past, future], ignore_index=True)
    merged = (merged.dropna(subset=["Fecha"])
                    .sort_values("Fecha")
                    .drop_duplicates("Fecha", keep="last")
                    .reset_index(drop=True))
    return merged

# ===================== Modelo (fallback interno) =====================
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
        self.input_min = np.array([1.0, 7.7, -3.5, 0.0], dtype=float)     # [JD, TMAX, TMIN, Prec]
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
        y  = self._tansig(z2)                  # [-1..1]
        emerrel01 = (y + 1.0) / 2.0            # [0..1]
        acc = np.cumsum(emerrel01)
        emeac01 = acc / 8.05                   # denominador de referencia
        emerrel = np.diff(emeac01, prepend=0.0)
        return emerrel, emeac01

# ===================== UI: encabezado =====================
st.title("BORDE2025 (histórico) + meteo_history (pronóstico) — Fusión sin reindex")
st.caption("La app usa únicamente las filas existentes en tus CSV. No inventa días.")

# ===================== CARGA =====================
hist = load_borde()
fcst = load_pron()

colA, colB = st.columns(2)
with colA:
    st.markdown("**BORDE2025 (histórico)**")
    if hist.empty:
        st.error("No se pudo leer el histórico o faltan columnas mínimas.")
    else:
        st.success(f"Histórico: {hist['Fecha'].min().date()} → {hist['Fecha'].max().date()} · {len(hist)} fila(s)")
with colB:
    st.markdown("**meteo_history (pronóstico)**")
    if fcst.empty:
        st.warning("No se pudo leer meteo_history.csv (se usará sólo histórico).")
    else:
        st.info(f"Pronóstico: {fcst['Fecha'].min().date()} → {fcst['Fecha'].max().date()} · {len(fcst)} fila(s)")

merged = merge_hist_forecast(hist, fcst)
if merged.empty:
    st.stop()

st.success(f"Serie fusionada: {merged['Fecha'].min().date()} → {merged['Fecha'].max().date()} · {len(merged)} fila(s)")

# ===================== Limpieza / Imputación (opcional) =====================
st.sidebar.markdown("### Calidad de datos")
IMPUTACION_CAUTA = st.sidebar.toggle("Imputación cauta (ffill/bfill 1 día en TMAX/TMIN; Prec=0 si falta)", value=False)
DROP_NAN_STRICT  = st.sidebar.toggle("Descartar filas con NaN críticos (Fecha/TMAX/TMIN)", value=True)
EXTRA_DROP_ALL   = st.sidebar.toggle("Descartar filas con NaN en cualquier core (Fecha/TMAX/TMIN/Prec/Julian_days)", value=False)

def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if IMPUTACION_CAUTA:
        for c in ["TMAX","TMIN"]:
            if c in df.columns:
                df[c] = df[c].ffill(limit=1).bfill(limit=1)
        if "Prec" in df.columns:
            df["Prec"] = df["Prec"].fillna(0)
    if DROP_NAN_STRICT:
        df = df.dropna(subset=["Fecha","TMAX","TMIN"])
        df["Prec"] = df["Prec"].fillna(0)
    if EXTRA_DROP_ALL:
        df = df.dropna(subset=[c for c in ["Fecha","TMAX","TMIN","Prec","Julian_days"] if c in df.columns])
    return df

merged = apply_cleaning(merged)

# 🔎 Filas con NaN tras limpieza
with st.expander("🔎 Filas con NaN tras limpieza"):
    core_cols = [c for c in ["Fecha","TMAX","TMIN","Prec","Julian_days"] if c in merged.columns]
    nan_mask = merged[core_cols].isna().any(axis=1) if core_cols else None
    if nan_mask is not None and nan_mask.any():
        st.warning(f"Se detectaron {int(nan_mask.sum())} fila(s) con NaN en columnas core.")
        st.dataframe(merged.loc[nan_mask].head(200), use_container_width=True)
        st.download_button(
            "Descargar filas con NaN (CSV)",
            merged.loc[nan_mask, core_cols].to_csv(index=False).encode("utf-8"),
            "filas_con_nan_post_limpieza.csv",
            "text/csv"
        )
    else:
        st.success("No hay NaN en columnas core tras la limpieza.")

# ===================== Temporada (JD base y recorte) =====================
st.sidebar.markdown("### Temporada")
# Base por defecto: primer día del pronóstico si existe, si no, primer día de la serie fusionada
try:
    _default_base = pd.to_datetime(fcst["Fecha"].min()).date() if not fcst.empty else pd.to_datetime(merged["Fecha"].min()).date()
except Exception:
    _default_base = pd.to_datetime(merged["Fecha"].min()).date()

season_base_date = st.sidebar.date_input("Inicio de temporada (JD=1)", value=_default_base)
CROP_TO_148 = st.sidebar.toggle("Recortar a máximo JD=148 (recomendado)", value=True)

# Recalcular JD relativo a la base elegida
season_base_ts = pd.Timestamp(season_base_date)
merged["Julian_days"] = (pd.to_datetime(merged["Fecha"]) - season_base_ts).dt.days + 1

if CROP_TO_148:
    before_len = len(merged)
    merged = merged[(merged["Julian_days"] >= 1) & (merged["Julian_days"] <= 148)].copy()
    after_len = len(merged)
    if after_len < before_len:
        st.info(f"Se recortaron {before_len - after_len} fila(s) fuera de JD 1..148 para ajustarse al rango del modelo.")
else:
    out_of_range = ((merged["Julian_days"] < 1) | (merged["Julian_days"] > 148)).sum()
    if out_of_range > 0:
        st.warning(f"Hay {int(out_of_range)} fila(s) con JD fuera de 1..148; el modelo las clippea y el desempeño puede degradarse.")

# ===================== 🛠 Depuración de serie de entrada =====================
with st.expander("🛠 Depuración de serie de entrada"):
    _df_dbg = merged.copy().sort_values("Fecha").reset_index(drop=True)
    st.write({
        "primera_fecha": str(_df_dbg["Fecha"].min().date()) if len(_df_dbg)>0 else None,
        "ultima_fecha":  str(_df_dbg["Fecha"].max().date()) if len(_df_dbg)>0 else None,
        "total_filas":   int(len(_df_dbg)),
    })
    st.markdown("**Últimas 10 fechas efectivas**")
    st.table(_df_dbg[["Fecha"]].tail(10))

# ===================== Modelo =====================
if merged.empty:
    st.error("No hay filas disponibles tras limpieza/temporada. Ajustá los toggles o revisá los CSV.")
    st.stop()

X = merged[["Julian_days","TMAX","TMIN","Prec"]].to_numpy(float)
model = PracticalANNModel()
emerrel, emeac01 = model.predict(X)

# Resultados DataFrame
pred = pd.DataFrame({
    "Fecha": merged["Fecha"].to_numpy(),
    "Julian_days": merged["Julian_days"].to_numpy(),
    "EMERREL(0-1)": np.nan_to_num(np.asarray(emerrel, dtype=float), nan=0.0),
    "EMEAC(0-1)": np.nan_to_num(np.asarray(emeac01, dtype=float), nan=0.0),
})
pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()
pred["EMEAC(%)"] = (pred["EMEAC(0-1)"] * 100).clip(0, 100)

# Clasificación simple
THR_BAJO_MEDIO = 0.02
THR_MEDIO_ALTO = 0.079
def nivel(v):
    if v < THR_BAJO_MEDIO: return "Bajo"
    elif v <= THR_MEDIO_ALTO: return "Medio"
    else: return "Alto"
pred["Nivel"] = pred["EMERREL(0-1)"].apply(nivel)
iconos = {"Bajo":"🟢 Bajo", "Medio":"🟠 Medio", "Alto":"🔴 Alto"}
colores = pred["Nivel"].map({"Bajo":"#2ca02c","Medio":"#ff7f0e","Alto":"#d62728"}).fillna("#808080")

# ===================== Grafico EMERREL =====================
st.subheader("EMERREL diario (barras) + MA5 (línea)")
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

# ===================== Grafico EMEAC =====================
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

# ===================== QA de consistencia =====================
with st.expander("🔍 QA de consistencia EMERREL/EMEAC"):
    emerrel_ok_num = pd.api.types.is_numeric_dtype(pred["EMERREL(0-1)"])
    emeac_ok_num   = pd.api.types.is_numeric_dtype(pred["EMEAC(0-1)"])
    emerrel_no_nan = not pd.isna(pred["EMERREL(0-1)"]).any()
    emeac_no_nan   = not pd.isna(pred["EMEAC(0-1)"]).any()

    emeac_monot    = pred["EMEAC(0-1)"].is_monotonic_increasing
    emeac_bounds   = float(pred["EMEAC(0-1)"].min()) >= 0.0 and float(pred["EMEAC(0-1)"].max()) <= 1.0 + 1e-9
    emerrel_nonneg = (pred["EMERREL(0-1)"] >= -1e-9).mean() >= 0.99

    acc_from_emerrel = pred["EMERREL(0-1)"].cumsum()
    residual = (pred["EMEAC(0-1)"] - acc_from_emerrel).astype(float)
    rmse = float((residual**2).mean()**0.5) if len(residual) else 0.0

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Tipos/NaN**")
        st.write({
            "emerrel_numeric": emerrel_ok_num,
            "emeac_numeric":   emeac_ok_num,
            "no_nan_emerrel":  emerrel_no_nan,
            "no_nan_emeac":    emeac_no_nan,
        })
    with col2:
        st.write("**Invariantes**")
        st.write({
            "emeac_monotonic_non_decreasing": emeac_monot,
            "emeac_bounds_0_1":               emeac_bounds,
            "emerrel_non_negative_mostly":    emerrel_nonneg,
            "consistency_emerrel_to_emeac_RMSE": rmse,
        })

    if not (emerrel_ok_num and emeac_ok_num and emerrel_no_nan and emeac_no_nan and emeac_monot and emeac_bounds and emerrel_nonneg):
        st.warning("⚠️ Algún check falló. Revisá NaN/tipos en BORDE2025/meteo_history o activá imputación cauta/descartes.")

    if st.button("Exportar auditoría (CSV)"):
        audit = pred.copy()
        audit["EMERREL_cumsum_recon"] = acc_from_emerrel
        st.download_button(
            "Descargar auditoría",
            audit.to_csv(index=False).encode("utf-8"),
            "auditoria_predweem_merge.csv",
            "text/csv",
            key="dl_audit_csv"
        )