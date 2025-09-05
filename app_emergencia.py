# app_emergencia.py — AVEFA (lockdown + empalme histórico adjunto 01-ene-2025 → 03-sep-2025 + futuro público + MA5 sombreada)
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path
import plotly.graph_objects as go
from typing import Callable, Any

# =================== LOCKDOWN UI ===================
st.set_page_config(
    page_title="Predicción de Emergencia Agrícola AVEFA",
    layout="wide",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header [data-testid="stToolbar"] {visibility: hidden;}
    .viewerBadge_container__1QSob {visibility: hidden;}
    .st-emotion-cache-9aoz2h {visibility: hidden;}
    .stAppDeployButton {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# =================== Utilidades de error seguro ===================
def safe_run(fn: Callable[[], Any], user_msg: str):
    try:
        return fn()
    except Exception:
        st.error(user_msg)
        return None

# ====================== Config pesos ======================
GITHUB_BASE_URL = "https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main"
FNAME_IW   = "IW.npy"
FNAME_BIW  = "bias_IW.npy"
FNAME_LW   = "LW.npy"
FNAME_BOUT = "bias_out.npy"

# ====================== Umbrales EMERREL ======================
THR_BAJO_MEDIO = 0.015
THR_MEDIO_ALTO = 0.05

# ====================== Umbrales EMEAC ======================
EMEAC_MIN_DEN = 3.0
EMEAC_ADJ_DEN = 3.5
EMEAC_MAX_DEN = 4.0

# ====================== Colores por nivel ======================
COLOR_MAP = {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
COLOR_FALLBACK = "#808080"

# ====================== Utilidades de red/archivos ======================
def _fetch_bytes(url: str, timeout: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (HTTPError, URLError):
        raise RuntimeError("No se pudo descargar el recurso remoto.")
    except Exception:
        raise RuntimeError("Error descargando recurso remoto.")

@st.cache_data(ttl=1800)
def load_npy_from_fixed(filename: str) -> np.ndarray:
    url = f"{GITHUB_BASE_URL}/{filename}"
    raw = _fetch_bytes(url)
    return np.load(BytesIO(raw), allow_pickle=False)

@st.cache_data(ttl=900)
def load_public_csv():
    urls = [
        "https://PREDWEEM.github.io/ANN/meteo_daily.csv",
        "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/meteo_daily.csv"
    ]
    for url in urls:
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"])
            req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
            if not req.issubset(df.columns):
                continue
            return df.sort_values("Fecha").reset_index(drop=True)
        except Exception:
            continue
    raise RuntimeError("No se pudo cargar el CSV público.")

def validar_columnas_meteo(df: pd.DataFrame):
    req = {"Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    return (len(faltan) == 0, "" if not faltan else f"Faltan columnas: {', '.join(sorted(faltan))}")

def obtener_colores(niveles: pd.Series):
    return niveles.map(COLOR_MAP).fillna(COLOR_FALLBACK).to_numpy()

def _sanitize_meteo(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Julian_days", "TMAX", "TMIN", "Prec"]
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[cols] = df[cols].interpolate(limit_direction="both")
    df["Julian_days"] = df["Julian_days"].clip(1, 366)
    df["Prec"] = df["Prec"].clip(lower=0)
    m = df["TMAX"] < df["TMIN"]
    if m.any():
        df.loc[m, ["TMAX", "TMIN"]] = df.loc[m, ["TMIN", "TMAX"]].values
    return df

# ====================== PERSISTENCIA LOCAL (CSV) ======================
LOCAL_HISTORY_PATH = st.secrets.get("LOCAL_HISTORY_PATH", "avefa_history_local.csv")
FREEZE_HISTORY = bool(st.secrets.get("FREEZE_HISTORY", False))  # valor por defecto

def _normalize_like_hist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    lower_map = {c.lower(): c for c in out.columns}
    def pick(*cands):
        for c in cands:
            if c in lower_map:
                return lower_map[c]
        return None
    mapping = {
        (pick("fecha","date") or "Fecha"): "Fecha",
        (pick("julian_days","julianday","julian","doy","dia_juliano") or "Julian_days"): "Julian_days",
        (pick("tmax","t_max","tx") or "TMAX"): "TMAX",
        (pick("tmin","t_min","tn") or "TMIN"): "TMIN",
        (pick("prec","ppt","precip","lluvia","mm","prcp") or "Prec"): "Prec",
    }
    out = out.rename(columns=mapping)
    if "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce")
    if "Julian_days" not in out.columns and "Fecha" in out.columns:
        out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear
    if "Fecha" not in out.columns and "Julian_days" in out.columns:
        year = pd.Timestamp.now().year
        out["Fecha"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(pd.to_numeric(out["Julian_days"], errors="coerce") - 1, unit="D")
    for c in ["TMAX","TMIN","Prec","Julian_days"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    req = {"Fecha","Julian_days","TMAX","TMIN","Prec"}
    if not req.issubset(out.columns):
        if "Fecha" in out.columns:
            out = out.dropna(subset=["Fecha"])
        return out.sort_values("Fecha").reset_index(drop=True)
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear
    return out[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def _load_local_history(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    try:
        df = pd.read_csv(p)
    except Exception:
        try:
            df = pd.read_excel(p)
        except Exception:
            return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    return _normalize_like_hist(df)

def _save_local_history(path: str, df_hist: pd.DataFrame) -> None:
    try:
        cols = ["Fecha","Julian_days","TMAX","TMIN","Prec"]
        to_write = df_hist[cols].copy() if set(cols).issubset(df_hist.columns) else df_hist.copy()
        to_write.to_csv(path, index=False, date_format="%Y-%m-%d")
    except Exception:
        pass

def _union_histories(df_prev: pd.DataFrame, df_new: pd.DataFrame, freeze_existing: bool = False) -> pd.DataFrame:
    prev = _normalize_like_hist(df_prev)
    new  = _normalize_like_hist(df_new)
    if prev.empty and new.empty:
        return prev
    if prev.empty:
        base = new
    elif new.empty:
        base = prev
    else:
        concat = pd.concat([prev, new], ignore_index=True)
        keep_mode = "first" if freeze_existing else "last"
        base = (concat.dropna(subset=["Fecha"])
                     .sort_values("Fecha")
                     .drop_duplicates(subset=["Fecha"], keep=keep_mode)
                     .reset_index(drop=True))
    base["Fecha"] = pd.to_datetime(base["Fecha"], errors="coerce")
    base = base.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    base["Julian_days"] = base["Fecha"].dt.dayofyear
    for c in ["TMAX","TMIN","Prec"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    return base

# ====================== Modelo ======================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = float(bias_out)
        # Orden de entrada: [Julian_days, TMIN, TMAX, Prec]
        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)
        self._den = np.maximum(self.input_max - self.input_min, 1e-9)

    def tansig(self, x): return np.tanh(x)

    def normalize_input(self, X):
        Xc = np.clip(X, self.input_min, self.input_max)
        return 2 * (Xc - self.input_min) / self._den - 1

    def denormalize_output(self, y, ymin=-1, ymax=1):
        return (y - ymin) / (ymax - ymin)

    def predict(self, X_real, thr_bajo_medio=THR_BAJO_MEDIO, thr_medio_alto=THR_MEDIO_ALTO):
        Xn = self.normalize_input(X_real)
        z1 = Xn @ self.IW + self.bias_IW
        a1 = self.tansig(z1)
        LW2 = self.LW.reshape(1, -1) if self.LW.ndim == 1 else self.LW
        z2 = (a1 @ LW2.T).ravel() + self.bias_out
        y  = self.tansig(z2)
        y  = self.denormalize_output(y)   # [0,1]
        ac = np.cumsum(y) / 8.05
        diff = np.diff(ac, prepend=0)
        niveles = np.where(diff <= thr_bajo_medio, "Bajo",
                   np.where(diff <= thr_medio_alto, "Medio", "Alto"))
        return pd.DataFrame({"EMERREL(0-1)": diff, "Nivel_Emergencia_relativa": niveles})

# ====================== UI ======================
st.title("Predicción de Emergencia Agrícola AVEFA")

st.sidebar.header("Meteo")
st.sidebar.caption("Histórico adjunto 01-ene-2025 → 03-sep-2025 + futuro del CSV público.")

# <<< NUEVO: opción de congelar en el menú lateral >>>
FREEZE_HISTORY = st.sidebar.checkbox(
    "Congelar histórico local (no sobrescribir)",
    value=FREEZE_HISTORY,
    help="Si está activado, al guardar el histórico local se conservan los valores ya guardados para cada fecha."
)

if st.sidebar.button("Limpiar caché"):
    st.cache_data.clear()

# --- Cargar pesos ---
def _cargar_pesos():
    IW       = load_npy_from_fixed(FNAME_IW)
    bias_IW  = load_npy_from_fixed(FNAME_BIW)
    LW       = load_npy_from_fixed(FNAME_LW)
    bout     = load_npy_from_fixed(FNAME_BOUT)

    if LW.ndim == 1:
        LW = LW.reshape(1, -1)
    if np.ndim(bout) == 0:
        bias_out = float(bout)
    else:
        bias_out = float(np.ravel(bout)[0])

    assert IW.shape[0] == 4, "Dimensiones de IW inválidas (esperado 4×N)."
    assert bias_IW.shape[0] == IW.shape[1], "bias_IW no coincide con N neuronas ocultas."
    assert LW.shape == (1, IW.shape[1]), "Dimensiones de LW inválidas (esperado 1×N)."

    return IW, bias_IW, LW, bias_out

pesos = safe_run(_cargar_pesos, "No se pudieron cargar los archivos del modelo.")
if pesos is None: st.stop()
IW, bias_IW, LW, bias_out = pesos
modelo = PracticalANNModel(IW, bias_IW, LW, float(bias_out))

# ====================== EMPALME: histórico adjunto + futuro público ======================
HIST_START = pd.Timestamp(2025, 1, 1)
HIST_END   = pd.Timestamp(2025, 9, 3)   # inclusive

def _normalize_meteo_generic(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None
    mapping = {
        (pick("fecha","date") or "Fecha"): "Fecha",
        (pick("julian_days","julianday","julian","doy","dia_juliano") or "Julian_days"): "Julian_days",
        (pick("tmax","t_max","tx") or "TMAX"): "TMAX",
        (pick("tmin","t_min","tn") or "TMIN"): "TMIN",
        (pick("prec","ppt","precip","lluvia","mm","prcp") or "Prec"): "Prec",
    }
    df = df.rename(columns=mapping)
    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    if "Julian_days" not in df.columns and "Fecha" in df.columns:
        df["Julian_days"] = pd.to_datetime(df["Fecha"]).dt.dayofyear
    if "Fecha" not in df.columns and "Julian_days" in df.columns:
        df["Fecha"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(
            pd.to_numeric(df["Julian_days"], errors="coerce") - 1, unit="D"
        )
    for c in ["Julian_days","TMAX","TMIN","Prec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    df["Julian_days"] = df["Fecha"].dt.dayofyear
    if "Prec" in df.columns:
        df["Prec"] = df["Prec"].clip(lower=0)
    if {"TMAX","TMIN"}.issubset(df.columns):
        m = df["TMAX"] < df["TMIN"]
        if m.any():
            df.loc[m, ["TMAX","TMIN"]] = df.loc[m, ["TMIN","TMAX"]].values
    return df[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def _load_attached_history() -> pd.DataFrame:
    up = st.file_uploader(
        "Subí el HISTÓRICO (CSV/XLSX) para 2025-01-01 → 2025-09-03",
        type=["csv","xlsx"], accept_multiple_files=False, key="hist_attach"
    )
    df_hist = pd.DataFrame()
    if up is not None:
        try:
            if up.name.lower().endswith(".xlsx"):
                df_hist = pd.read_excel(up)
            else:
                df_hist = pd.read_csv(up)
        except Exception:
            st.error("No se pudo leer el archivo adjunto. Verificá formato/columnas.")
            df_hist = pd.DataFrame()
    else:
        try:
            p = Path("/mnt/data/BORDE2025.csv")
            if p.exists():
                df_hist = pd.read_csv(p)
        except Exception:
            pass

    if df_hist.empty:
        st.warning("No se detectó histórico adjunto; se continuará solo con el CSV público.")
        return df_hist

    df_hist = _normalize_meteo_generic(df_hist)
    m = (df_hist["Fecha"] >= HIST_START) & (df_hist["Fecha"] <= HIST_END)
    df_hist = df_hist.loc[m].copy().reset_index(drop=True)

    if df_hist.empty:
        st.error("El histórico adjunto no tiene filas dentro de 2025-01-01 → 2025-09-03.")
        return df_hist

    # Interpolación si hay NaN
    faltantes = [c for c in ["Julian_days","TMAX","TMIN","Prec"] if df_hist[c].isna().any()]
    if faltantes:
        st.warning(f"El histórico adjunto tiene NaN en: {', '.join(faltantes)}. Se interpolará.")
        df_hist[["Julian_days","TMAX","TMIN","Prec"]] = df_hist[["Julian_days","TMAX","TMIN","Prec"]].interpolate(
            limit_direction="both"
        )

    st.success(f"Histórico adjunto OK: {df_hist['Fecha'].min().date()} → {df_hist['Fecha'].max().date()} "
               f"({len(df_hist)} filas)")
    return df_hist

def _leer_public_csv_solo_futuro():
    df_pub = load_public_csv()
    df_pub = _sanitize_meteo(df_pub)
    if "Fecha" not in df_pub.columns or not np.issubdtype(df_pub["Fecha"].dtype, np.datetime64):
        df_pub["Fecha"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df_pub["Julian_days"] - 1, unit="D")
    df_pub = df_pub.sort_values("Fecha").reset_index(drop=True)
    df_future = df_pub.loc[df_pub["Fecha"] > HIST_END].copy()
    return df_future

df_hist_attached = _load_attached_history()
df_future_pub = safe_run(_leer_public_csv_solo_futuro, "No se pudo cargar el CSV público.")

dfs = []
if df_hist_attached is not None and not df_hist_attached.empty:
    base_hist = df_hist_attached.copy()
else:
    base_hist = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

if df_future_pub is not None:
    df_future_pub = df_future_pub.loc[df_future_pub["Fecha"] > HIST_END]
else:
    df_future_pub = pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

df_empalmado = pd.concat([base_hist, df_future_pub], ignore_index=True)
df_empalmado = df_empalmado.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)

# Persistencia local combinada con FREEZE_HISTORY
try:
    df_prev_local = _load_local_history(LOCAL_HISTORY_PATH)
    df_union = _union_histories(df_prev_local, df_empalmado, freeze_existing=FREEZE_HISTORY)
    _save_local_history(LOCAL_HISTORY_PATH, df_union)
    df_empalmado = df_union
except Exception:
    pass

if df_empalmado.empty:
    st.stop()
else:
    dfs = [("Histórico adjunto + Pronóstico público", df_empalmado)]

if df_future_pub is not None and df_future_pub.empty:
    st.info("El CSV público no contiene días posteriores a 2025-09-03. Solo se mostrará el histórico adjunto.")
else:
    st.success(f"Empalme OK. Futuro detectado: {len(df_empalmado.loc[df_empalmado['Fecha'] > HIST_END])} día(s) posteriores a 2025-09-03.")

# ====================== Procesamiento y visualización ======================
for nombre, df in dfs:
    ok, msg = validar_columnas_meteo(df)
    if not ok:
        st.warning(f"{nombre}: {msg}")
        continue

    df = df.sort_values("Julian_days").reset_index(drop=True)
    # Asegurar 'Fecha'
    if "Fecha" not in df.columns or not np.issubdtype(df["Fecha"].dtype, np.datetime64):
        year = pd.Timestamp.now().year
        df["Fecha"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")

    X_real = df[["Julian_days", "TMIN", "TMAX", "Prec"]].to_numpy(float)
    fechas = pd.to_datetime(df["Fecha"])

    pred = modelo.predict(X_real, thr_bajo_medio=THR_BAJO_MEDIO, thr_medio_alto=THR_MEDIO_ALTO)
    pred["Fecha"] = fechas
    pred["Julian_days"] = df["Julian_days"]
    pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
    pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

    pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
    pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
    pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / EMEAC_ADJ_DEN
    for col in ["EMEAC (0-1) - mínimo", "EMEAC (0-1) - máximo", "EMEAC (0-1) - ajustable"]:
        pred[col.replace("(0-1)", "(%)")] = (pred[col] * 100).clip(0, 100)

    # ---- Ventana 1/feb → 1/nov para gráficos (con fallback) ----
    years = pred["Fecha"].dt.year.unique()
    yr = int(years[0]) if len(years) == 1 else int(st.sidebar.selectbox("Año (reinicio 1/feb → 1/nov)", sorted(years), key=f"year_select_{nombre}"))
    fi = pd.Timestamp(year=yr, month=2, day=1)
    ff = pd.Timestamp(year=yr, month=11, day=1)
    m = (pred["Fecha"] >= fi) & (pred["Fecha"] <= ff)
    pred_vis = pred.loc[m].copy()

    rango_personalizado = False
    if pred_vis.empty:
        pred_vis = pred.copy()
        fi, ff = pred_vis["Fecha"].min(), pred_vis["Fecha"].max()
        rango_personalizado = True
        st.info(f"{nombre}: sin datos entre 1/feb y 1/nov; mostrando rango real disponible: {fi.date()} → {ff.date()}.")

    pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
    pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_MIN_DEN
    pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_MAX_DEN
    pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_ADJ_DEN
    for col in ["EMEAC (0-1) - mínimo (rango)", "EMEAC (0-1) - máximo (rango)", "EMEAC (0-1) - ajustable (rango)"]:
        pred_vis[col.replace("(0-1)", "(%)")] = (pred_vis[col] * 100).clip(0, 100)

    colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

    # ====== FIGURA: EMERREL diario (con MA5 sombreada) ======
    st.subheader("EMERGENCIA RELATIVA DIARIA")
    fig_er = go.Figure()
    fig_er.add_bar(
        x=pred_vis["Fecha"],
        y=pred_vis["EMERREL(0-1)"],
        marker=dict(color=colores_vis.tolist()),
        customdata=pred_vis["Nivel_Emergencia_relativa"].map(
            {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}
        ),
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
        name="EMERREL (0-1)"
    )
    # Línea MA5
    fig_er.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5"],
        mode="lines", line=dict(width=2),
        name="Media móvil 5 días",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
    ))
    # Sombreado tenue bajo MA5
    fig_er.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMERREL_MA5"],
        mode="lines", line=dict(width=0),
        fill="tozeroy", fillcolor="rgba(65,105,225,0.15)",
        hoverinfo="skip", showlegend=False
    ))
    # Niveles de referencia
    low_thr = float(THR_BAJO_MEDIO); med_thr = float(THR_MEDIO_ALTO)
    fig_er.add_trace(go.Scatter(x=[fi, ff], y=[low_thr, low_thr],
        mode="lines", line=dict(color=COLOR_MAP["Bajo"], dash="dot"),
        name=f"Bajo (≤ {low_thr:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[fi, ff], y=[med_thr, med_thr],
        mode="lines", line=dict(color=COLOR_MAP["Medio"], dash="dot"),
        name=f"Medio (≤ {med_thr:.3f})", hoverinfo="skip"))
    fig_er.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color=COLOR_MAP["Alto"], dash="dot"),
        name=f"Alto (> {med_thr:.3f})", hoverinfo="skip"))
    fig_er.update_layout(
        xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
        hovermode="x unified", legend_title="Referencias", height=650
    )
    fig_er.update_xaxes(range=[fi, ff], dtick="D1" if (ff-fi).days <= 31 else "M1",
                        tickformat="%d-%b" if (ff-fi).days <= 31 else "%b")
    fig_er.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

    # ====== FIGURA: EMERGENCIA acumulada ======
    st.subheader("EMERGENCIA ACUMULADA DIARIA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo (rango)"],
        mode="lines", line=dict(width=0), name="Máximo (reiniciado)",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo (rango)"],
        mode="lines", line=dict(width=0), fill="tonexty", name="Mínimo (reiniciado)",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - ajustable (rango)"],
        mode="lines", line=dict(width=2.5),
        name=f"Umbral ajustable (/{EMEAC_ADJ_DEN:.2f})",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - mínimo (rango)"],
        mode="lines", line=dict(dash="dash", width=1.5),
        name=f"Umbral mínimo (/{EMEAC_MIN_DEN:.2f})",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=pred_vis["Fecha"], y=pred_vis["EMEAC (%) - máximo (rango)"],
        mode="lines", line=dict(dash="dash", width=1.5),
        name=f"Umbral máximo (/{EMEAC_MAX_DEN:.2f})",
        hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
    ))
    for nivel in [25, 50, 75, 90]:
        try:
            fig.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")
        except Exception:
            fig.add_trace(go.Scatter(x=[pred_vis["Fecha"].min(), pred_vis["Fecha"].max()],
                                     y=[nivel, nivel], mode="lines", line=dict(dash="dash"), showlegend=False))
    fig.update_layout(
        xaxis_title="Fecha", yaxis_title="EMEAC (%)",
        yaxis=dict(range=[0, 100]), hovermode="x unified",
        legend_title="Referencias", height=600
    )
    fig.update_xaxes(range=[fi, ff], dtick="D1" if (ff-fi).days <= 31 else "M1",
                     tickformat="%d-%b" if (ff-fi).days <= 31 else "%b")
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # ====== TABLA: Resultados en el rango gráfico ======
    rango_txt = f"{fi.date()} → {ff.date()}" if not (pred_vis["Fecha"].min() == fi and pred_vis["Fecha"].max() == ff) else "1/feb → 1/nov"
    st.subheader(f"Resultados ({rango_txt}) - {nombre}")
    col_emeac = "EMEAC (%) - ajustable (rango)"
    nivel_icono = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}
    tabla_rango = pred_vis[["Fecha","Julian_days","Nivel_Emergencia_relativa",col_emeac]].copy()
    tabla_rango["Nivel_Emergencia_relativa"] = tabla_rango["Nivel_Emergencia_relativa"].map(nivel_icono)
    tabla_rango = tabla_rango.rename(columns={"Nivel_Emergencia_relativa":"Nivel de EMERREL", col_emeac:"EMEAC (%)"})
    st.dataframe(tabla_rango, use_container_width=True)

    csv_buf = StringIO(); tabla_rango.to_csv(csv_buf, index=False)
    st.download_button(
        f"Descargar resultados (rango) - {nombre}",
        data=csv_buf.getvalue(),
        file_name=f"{nombre}_resultados_rango.csv",
        mime="text/csv"
    )

# ====================== SELF-TEST (opcional) ======================
st.sidebar.markdown("---")
if st.sidebar.button("🔎 Autotest del modelo (6 días sintéticos)"):
    base = pd.Timestamp(pd.Timestamp.now().year, 9, 1)
    fechas_t = pd.date_range(base, periods=6, freq="D")
    df_test = pd.DataFrame({
        "Fecha": fechas_t,
        "Julian_days": fechas_t.dayofyear,
        "TMIN": [2.0, 3.5, 4.0, 5.0, 6.0, 7.5],
        "TMAX": [12.0, 14.5, 15.0, 16.0, 17.0, 18.5],
        "Prec": [0.0, 2.5, 0.0, 10.0, 3.0, 0.0],
    })
    X_test = df_test[["Julian_days", "TMIN", "TMAX", "Prec"]].to_numpy(float)
    out = modelo.predict(X_test, thr_bajo_medio=THR_BAJO_MEDIO, thr_medio_alto=THR_MEDIO_ALTO).copy()
    out["Fecha"] = df_test["Fecha"]
    out["Julian_days"] = df_test["Julian_days"]
    out["EMERREL acumulado"] = out["EMERREL(0-1)"].cumsum()

    assert out["EMERREL(0-1)"].notna().all(), "EMERREL contiene NaN."
    assert (out["EMERREL(0-1)"] >= 0).all(), "EMERREL < 0 detectado."
    assert (out["EMERREL(0-1)"] <= 1).all(), "EMERREL > 1 detectado."

    st.success("Autotest OK: inferencia realizada y valores en rango [0, 1].")

    try:
        fig_t = go.Figure()
        col_map = {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
        colores = out["Nivel_Emergencia_relativa"].map(col_map).fillna("#808080").tolist()
        fig_t.add_bar(x=out["Fecha"], y=out["EMERREL(0-1)"], marker=dict(color=colores), name="EMERREL(0-1)")
        fig_t.add_trace(go.Scatter(x=out["Fecha"], y=out["EMERREL(0-1)"].rolling(3, min_periods=1).mean(),
                                   mode="lines", line=dict(width=2), name="MA3"))
        fig_t.update_layout(xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
                            hovermode="x unified", height=450, title="Autotest – EMERREL diario")
        st.plotly_chart(fig_t, use_container_width=True, theme="streamlit")
    except Exception as e:
        st.info(f"Gráfico del autotest no crítico: {e}")
