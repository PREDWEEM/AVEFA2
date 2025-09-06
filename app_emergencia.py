# app_emergencia.py — AVEFA
# (lockdown + empalme histórico adjunto 01-ene-2025 → 03-sep-2025 + futuro público
#  + MA5 con relleno tricolor INTERNO (verde/amarillo/rojo, con opacidad) + botón Actualizar
#  + fix fechas dd/mm y DOY + lectura automática BORDE2025.csv desde GitHub)
import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path
import plotly.graph_objects as go
from typing import Callable, Any, List
import hashlib

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

# =================== Utilidades ===================
def safe_run(fn: Callable[[], Any], user_msg: str):
    try:
        return fn()
    except Exception:
        st.error(user_msg)
        return None

def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            st.warning("No pude forzar el rerun automáticamente. Volvé a ejecutar la app.")

# ====================== Config pesos ======================
GITHUB_BASE_URL = "https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main"
FNAME_IW   = "IW.npy"
FNAME_BIW  = "bias_IW.npy"
FNAME_LW   = "LW.npy"
FNAME_BOUT = "bias_out.npy"

# ====================== Umbrales EMERREL ======================
THR_BAJO_MEDIO = 0.01
THR_MEDIO_ALTO = 0.05

# ====================== Umbrales EMEAC ======================
EMEAC_MIN_DEN = 3.0
EMEAC_ADJ_DEN = 4.0
EMEAC_MAX_DEN = 5.0

# ====================== Colores por nivel (intensos) ======================
COLOR_MAP = {
    "Bajo":  "#00A651",  # verde intenso
    "Medio": "#FFC000",  # amarillo intenso
    "Alto":  "#E53935"   # rojo intenso
}
COLOR_FALLBACK = "#808080"

# ====================== Utilidades de red/archivos ======================
def _fetch_bytes(url: str, timeout: int = 20) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except (HTTPError, URLError):
        raise RuntimeError(f"No se pudo descargar el recurso remoto: {url}")
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
    last_err = None
    for url in urls:
        try:
            raw = _fetch_bytes(url)
            for use_sc, dec in [(True, ","), (True, "."), (False, ".")]:
                try:
                    bio = BytesIO(raw)
                    if use_sc:
                        df = pd.read_csv(bio, sep=";", decimal=dec, parse_dates=False)
                    else:
                        df = pd.read_csv(bio, decimal=dec, parse_dates=False)
                    if "Fecha" in df.columns:
                        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
                    req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
                    if not req.issubset(df.columns):
                        continue
                    for c in ["Julian_days", "TMAX", "TMIN", "Prec"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    return df.sort_values("Fecha").reset_index(drop=True)
                except Exception:
                    continue
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No se pudo cargar el CSV público. Último error: {last_err}")

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

# ====================== HISTÓRICO DESDE GITHUB ======================
HIST_CSV_URL_SECRET = st.secrets.get("HIST_CSV_URL", "").strip()
HIST_CSV_URLS: List[str] = [
    "https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main/BORDE2025.csv",
    "https://raw.githubusercontent.com/PREDWEEM/ANN/gh-pages/BORDE2025.csv",
    "https://PREDWEEM.github.io/ANN/BORDE2025.csv",
]

def _try_read_csv_semicolon_first(url: str) -> pd.DataFrame:
    raw = _fetch_bytes(url)
    try:
        df = pd.read_csv(BytesIO(raw), sep=";")
        if not df.empty:
            return df
    except Exception:
        pass
    try:
        return pd.read_csv(BytesIO(raw))
    except Exception as e:
        raise RuntimeError(f"No se pudo leer CSV desde {url} ({e})")

@st.cache_data(ttl=900)
def load_borde_from_github() -> pd.DataFrame:
    urls = []
    if HIST_CSV_URL_SECRET:
        urls.append(HIST_CSV_URL_SECRET)
    urls.extend(HIST_CSV_URLS)
    last_err = None
    for url in urls:
        try:
            df = _try_read_csv_semicolon_first(url)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No pude leer BORDE2025.csv desde GitHub. Último error: {last_err}")

# ====================== PERSISTENCIA LOCAL (CSV) ======================
LOCAL_HISTORY_PATH = st.secrets.get("LOCAL_HISTORY_PATH", "avefa_history_local.csv")
FREEZE_HISTORY = bool(st.secrets.get("FREEZE_HISTORY", False))

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
    if "Julian_days" in out.columns:
        out["Julian_days"] = pd.to_numeric(out["Julian_days"], errors="coerce")
        out["Fecha"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(out["Julian_days"] - 1, unit="D")
    elif "Fecha" in out.columns:
        out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce", dayfirst=True)
        out["Julian_days"] = pd.to_datetime(out["Fecha"]).dt.dayofyear
    else:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    for c in ["TMAX","TMIN","Prec"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
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
    base = base.dropna(subset=["Fecha"]).sort_values("Fecha").drop_duplicates(subset=["Fecha"], keep="last").reset_index(drop=True)
    base["Julian_days"] = base["Fecha"].dt.dayofyear
    for c in ["TMAX","TMIN","Prec"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base[["Julian_days","TMAX","TMIN","Prec"]] = base[["Julian_days","TMAX","TMIN","Prec"]].interpolate(limit_direction="both")
    m = base["TMAX"] < base["TMIN"]
    if m.any():
        base.loc[m, ["TMAX","TMIN"]] = base.loc[m, ["TMIN","TMAX"]].values
    base["Prec"] = base["Prec"].clip(lower=0)
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
st.sidebar.caption("Histórico BORDE2025.csv se carga automáticamente desde GitHub; futuro desde el CSV público.")

if "cache_bust" not in st.session_state:
    st.session_state.cache_bust = 0

col_a, col_b = st.sidebar.columns([1,1])
with col_a:
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.session_state.cache_bust += 1
        _safe_rerun()
with col_b:
    if st.button("🧹 Limpiar caché"):
        st.cache_data.clear()
        st.success("Caché limpiada. Volvé a correr o tocá 'Actualizar datos'.")

FREEZE_HISTORY = st.sidebar.checkbox(
    "Congelar histórico local (no sobrescribir)",
    value=FREEZE_HISTORY,
    help="Si está activado, al guardar el histórico local se conservan los valores ya guardados para cada fecha."
)

st.sidebar.markdown("---")
rango_opcion = st.sidebar.radio(
    "Rango para mostrar",
    ["1/feb → 1/nov", "Todo el empalme"],
    index=0,
    help="Esto solo afecta los gráficos y la tabla. El modelo SIEMPRE se ejecuta sobre todo el empalme."
)

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
if pesos is None:
    st.stop()
IW, bias_IW, LW, bias_out = pesos
modelo = PracticalANNModel(IW, bias_IW, LW, float(bias_out))

# ====================== EMPALME: histórico GitHub + futuro público ======================
HIST_START = pd.Timestamp(2025, 1, 1)
HIST_END   = pd.Timestamp(2025, 9, 3)   # inclusive

def _normalize_meteo_generic(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    df = df_in.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    def pick(*cands):
        for c in cands:
            if c in lower:
                return lower[c]
        return None
    c_fecha = pick("fecha","date","fech","día","dia","fch")
    c_doy   = pick("julian_days","julianday","julian","doy","dia_juliano","diajuliano","juliano","dayofyear","día juliano","dia juliano")
    c_tmax  = pick("tmax","t_max","tx","t máx","t max","t. max")
    c_tmin  = pick("tmin","t_min","tn","t mín","t min","t. min")
    c_prec  = pick("prec","ppt","precip","lluvia","mm","prcp","precipitacion","precipitación")
    mapping = {}
    if c_fecha: mapping[c_fecha] = "Fecha"
    if c_doy:   mapping[c_doy]   = "Julian_days"
    if c_tmax:  mapping[c_tmax]  = "TMAX"
    if c_tmin:  mapping[c_tmin]  = "TMIN"
    if c_prec:  mapping[c_prec]  = "Prec"
    if mapping:
        df = df.rename(columns=mapping)
    if "Julian_days" in df.columns:
        df["Julian_days"] = pd.to_numeric(df["Julian_days"], errors="coerce")
        df["Fecha"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df["Julian_days"] - 1, unit="D")
    elif "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
        df["Julian_days"] = pd.to_datetime(df["Fecha"]).dt.dayofyear
    else:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    for c in ["TMAX","TMIN","Prec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Prec" in df.columns:
        df["Prec"] = df["Prec"].clip(lower=0)
    if {"TMAX","TMIN"}.issubset(df.columns):
        m = df["TMAX"] < df["TMIN"]
        if m.any():
            df.loc[m, ["TMAX","TMIN"]] = df.loc[m, ["TMIN","TMAX"]].values
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
    for need in ["TMAX","TMIN","Prec"]:
        if need not in df.columns:
            df[need] = np.nan
    return df[["Fecha","Julian_days","TMAX","TMIN","Prec"]]

def _load_attached_history_from_github() -> pd.DataFrame:
    try:
        df_hist_raw = load_borde_from_github()
        st.success("BORDE2025.csv cargado automáticamente desde GitHub ✅")
        return _normalize_meteo_generic(df_hist_raw)
    except Exception as e:
        st.warning(f"No se pudo cargar BORDE2025.csv desde GitHub ({e}). Intentando fallback...")
        return pd.DataFrame()

def _load_attached_history_fallback() -> pd.DataFrame:
    up = st.file_uploader(
        "Subí el HISTÓRICO (CSV/XLSX) para 2025-01-01 → 2025-09-03 (solo si falla GitHub)",
        type=["csv","xlsx"], accept_multiple_files=False, key="hist_attach"
    )
    df_hist_raw = pd.DataFrame()
    if up is not None:
        try:
            if up.name.lower().endswith(".xlsx"):
                df_hist_raw = pd.read_excel(up)
            else:
                try:
                    df_hist_raw = pd.read_csv(up, sep=";")
                except Exception:
                    up.seek(0)
                    df_hist_raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"No se pudo leer el archivo adjunto ({e}).")
            return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    else:
        try:
            p = Path("/mnt/data/BORDE2025.csv")
            if p.exists():
                try:
                    df_hist_raw = pd.read_csv(p, sep=";")
                except Exception:
                    df_hist_raw = pd.read_csv(p)
        except Exception as e:
            st.info(f"No se pudo leer /mnt/data/BORDE2025.csv ({e}).")
            return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    if df_hist_raw.empty:
        return pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
    return _normalize_meteo_generic(df_hist_raw)

# Carga priorizando GitHub
df_hist_attached = _load_attached_history_from_github()
if df_hist_attached.empty:
    df_hist_attached = _load_attached_history_fallback()

# Acotar al rango histórico esperado
if not df_hist_attached.empty:
    m = (df_hist_attached["Fecha"] >= HIST_START) & (df_hist_attached["Fecha"] <= HIST_END)
    df_hist_attached = df_hist_attached.loc[m].copy().reset_index(drop=True)
    if df_hist_attached.empty:
        st.error("El histórico (GitHub/fallback) no tiene filas dentro de 2025-01-01 → 2025-09-03.")

# Verificación de huecos en todo el histórico esperado
if not df_hist_attached.empty:
    full_hist_range = pd.date_range(HIST_START, HIST_END, freq="D")
    present = pd.to_datetime(df_hist_attached["Fecha"]).dt.normalize().unique()
    missing = [d for d in full_hist_range if d.to_datetime64() not in present]
    if missing:
        st.warning("El histórico adjunto tiene huecos reales en 2025-01-01 → 2025-09-03: " +
                   ", ".join(pd.DatetimeIndex(missing).strftime("%d-%m").tolist()))
    else:
        st.info("Histórico (01-ene → 03-sep) completo ✅")

def _leer_public_csv_solo_futuro():
    df_pub = load_public_csv()
    df_pub = _sanitize_meteo(df_pub)
    if "Fecha" not in df_pub.columns or not np.issubdtype(df_pub["Fecha"].dtype, np.datetime64):
        df_pub["Fecha"] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df_pub["Julian_days"] - 1, unit="D")
    df_pub = df_pub.sort_values("Fecha").reset_index(drop=True)
    return df_pub.loc[df_pub["Fecha"] > HIST_END].copy()

df_future_pub = safe_run(_leer_public_csv_solo_futuro, "No se pudo cargar el CSV público.")

# Unión: GitHub (manda en 1-ene→3-sep) + público (>3-sep)
base_hist = df_hist_attached.copy() if (df_hist_attached is not None and not df_hist_attached.empty) \
           else pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])
df_future_pub = df_future_pub.loc[df_future_pub["Fecha"] > HIST_END] if df_future_pub is not None \
                else pd.DataFrame(columns=["Fecha","Julian_days","TMAX","TMIN","Prec"])

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

# --- Chequeo de continuidad en Agosto 2025 ---
aug_start = pd.Timestamp(2025, 8, 1)
aug_end   = pd.Timestamp(2025, 8, 31)
cal_aug = pd.date_range(aug_start, aug_end, freq="D")
fechas_emp = pd.to_datetime(df_empalmado["Fecha"]).dt.normalize().unique()
faltan_aug = [d for d in cal_aug if d.normalize().to_datetime64() not in fechas_emp]
if faltan_aug:
    st.warning(f"Faltan {len(faltan_aug)} fecha(s) de agosto en el empalme: " +
               ", ".join(d.strftime("%d-%m") for d in faltan_aug))
else:
    st.info("Agosto 2025 completo en el empalme ✅")

# ===== Hash del empalme y recálculo automático del modelo =====
def _df_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    key_cols = ["Fecha","Julian_days","TMAX","TMIN","Prec"]
    keep = [c for c in key_cols if c in df.columns]
    sub = df[keep].copy()
    if "Fecha" in sub.columns:
        sub["Fecha"] = pd.to_datetime(sub["Fecha"]).dt.strftime("%Y-%m-%d")
    for c in ["Julian_days","TMAX","TMIN","Prec"]:
        if c in sub.columns:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
    raw = sub.to_csv(index=False).encode("utf-8")
    return hashlib.md5(raw).hexdigest()

empalme_hash = _df_hash(df_empalmado)
if "last_empalme_hash" not in st.session_state:
    st.session_state.last_empalme_hash = None
if "pred_full_empalme" not in st.session_state:
    st.session_state.pred_full_empalme = None
recalcular_pred = (st.session_state.last_empalme_hash != empalme_hash)

def _recompute_pred(df_emp):
    X_all = df_emp[["Julian_days","TMIN","TMAX","Prec"]].to_numpy(float)
    pred_all_ = modelo.predict(X_all, thr_bajo_medio=THR_BAJO_MEDIO, thr_medio_alto=THR_MEDIO_ALTO)
    pred_all_["Fecha"] = pd.to_datetime(df_emp["Fecha"])
    pred_all_["Julian_days"] = df_emp["Julian_days"]
    pred_all_["EMERREL acumulado"] = pred_all_["EMERREL(0-1)"].cumsum()
    return pred_all_

if recalcular_pred or st.session_state.get("pred_full_empalme") is None:
    pred_all = _recompute_pred(df_empalmado)
    st.session_state.pred_full_empalme = pred_all
    st.session_state.last_empalme_hash = empalme_hash
else:
    pred_all = st.session_state.pred_full_empalme.copy()

# Avisos de futuro
futuros = int((df_empalmado["Fecha"] > HIST_END).sum())
if futuros == 0:
    st.info("Empalme OK. Por ahora no hay días posteriores a 2025-09-03 en el CSV público; se muestra solo el histórico de GitHub.")
else:
    st.success(f"Empalme OK. Futuro detectado: {futuros} día(s) posteriores a 2025-09-03.")

# ====================== Procesamiento y visualización ======================
nombre = "Histórico GitHub + Pronóstico público"
df = df_empalmado.copy()
ok, msg = validar_columnas_meteo(df)
if not ok:
    st.warning(f"{nombre}: {msg}")
    st.stop()

pred = pred_all.copy()

# EMEAC global (sobre TODO el empalme)
pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / EMEAC_ADJ_DEN
for col in ["EMEAC (0-1) - mínimo", "EMEAC (0-1) - máximo", "EMEAC (0-1) - ajustable"]:
    pred[col.replace("(0-1)", "(%)")] = (pred[col] * 100).clip(0, 100)

# ====== Rango de visualización ======
if rango_opcion == "Todo el empalme":
    pred_vis = pred.copy()
    fi, ff = pred_vis["Fecha"].min(), pred_vis["Fecha"].max()
    rango_txt = f"{fi.date()} → {ff.date()}"
    y_min = pred_vis["EMEAC (%) - mínimo"]
    y_max = pred_vis["EMEAC (%) - máximo"]
    y_adj = pred_vis["EMEAC (%) - ajustable"]
else:
    years = pred["Fecha"].dt.year.unique()
    yr = int(years[0]) if len(years) == 1 else int(st.sidebar.selectbox(
        "Año (reinicio 1/feb → 1/nov)", sorted(years), key=f"year_select_{nombre}"
    ))
    fi = pd.Timestamp(year=yr, month=2, day=1)
    ff = pd.Timestamp(year=yr, month=11, day=1)
    m = (pred["Fecha"] >= fi) & (pred["Fecha"] <= ff)
    pred_vis = pred.loc[m].copy()
    if pred_vis.empty:
        pred_vis = pred.copy()
        fi, ff = pred_vis["Fecha"].min(), pred_vis["Fecha"].max()
    pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
    pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_MIN_DEN
    pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_MAX_DEN
    pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_ADJ_DEN
    for col in ["EMEAC (0-1) - mínimo (rango)", "EMEAC (0-1) - máximo (rango)", "EMEAC (0-1) - ajustable (rango)"]:
        pred_vis[col.replace("(0-1)", "(%)")] = (pred_vis[col] * 100).clip(0, 100)
    rango_txt = "1/feb → 1/nov"
    y_min = pred_vis["EMEAC (%) - mínimo (rango)"]
    y_max = pred_vis["EMEAC (%) - máximo (rango)"]
    y_adj = pred_vis["EMEAC (%) - ajustable (rango)"]

# ====== Colores por nivel y MA5 ======
colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])
pred_vis["EMERREL_MA5"] = pred_vis["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

# ====== FIGURA: EMERGENCIA RELATIVA DIARIA (MA5 con relleno tricolor INTERNO, con opacidad) ======
st.subheader("EMERGENCIA RELATIVA DIARIA")
fig_er = go.Figure()

# Barras coloreadas por nivel
fig_er.add_bar(
    x=pred_vis["Fecha"],
    y=pred_vis["EMERREL(0-1)"],
    marker=dict(color=colores_vis.tolist()),
    customdata=pred_vis["Nivel_Emergencia_relativa"].map(
        {"Bajo": "🟢 Bajo", "Medio": "🟡 Medio", "Alto": "🔴 Alto"}
    ),
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
    name="EMERREL (0-1)"
)

# Área interna bajo MA5 segmentada (0→0.01 verde, 0.01→0.05 amarillo, 0.05→MA5 rojo) con opacidad
x = pred_vis["Fecha"]
ma = pred_vis["EMERREL_MA5"].clip(lower=0)
thr_low = float(THR_BAJO_MEDIO)   # 0.01
thr_med = float(THR_MEDIO_ALTO)   # 0.05

y0 = np.zeros(len(ma))
y1 = np.minimum(ma, thr_low)   # tope verde
y2 = np.minimum(ma, thr_med)   # tope amarillo
y3 = ma                        # tope rojo

# === Colores con opacidad suave ===
ALPHA = 0.28  # ajustá 0.20–0.35 a gusto
GREEN_RGBA  = f"rgba(0,166,81,{ALPHA})"
YELLOW_RGBA = f"rgba(255,192,0,{ALPHA})"
RED_RGBA    = f"rgba(229,57,53,{ALPHA})"

# Base 0
fig_er.add_trace(go.Scatter(
    x=x, y=y0, mode="lines",
    line=dict(width=0),
    hoverinfo="skip", showlegend=False
))
# Banda VERDE (hacia y0)
fig_er.add_trace(go.Scatter(
    x=x, y=y1, mode="lines",
    line=dict(width=0),
    fill="tonexty", fillcolor=GREEN_RGBA,
    hoverinfo="skip", showlegend=False, name="Zona baja (verde)"
))
# Baseline y1
fig_er.add_trace(go.Scatter(
    x=x, y=y1, mode="lines",
    line=dict(width=0),
    hoverinfo="skip", showlegend=False
))
# Banda AMARILLA (hacia y1)
fig_er.add_trace(go.Scatter(
    x=x, y=y2, mode="lines",
    line=dict(width=0),
    fill="tonexty", fillcolor=YELLOW_RGBA,
    hoverinfo="skip", showlegend=False, name="Zona media (amarillo)"
))
# Baseline y2
fig_er.add_trace(go.Scatter(
    x=x, y=y2, mode="lines",
    line=dict(width=0),
    hoverinfo="skip", showlegend=False
))
# Banda ROJA (hacia y2)
fig_er.add_trace(go.Scatter(
    x=x, y=y3, mode="lines",
    line=dict(width=0),
    fill="tonexty", fillcolor=RED_RGBA,
    hoverinfo="skip", showlegend=False, name="Zona alta (rojo)"
))

# Línea MA5
fig_er.add_trace(go.Scatter(
    x=x, y=ma, mode="lines",
    line=dict(width=2),
    name="Media móvil 5 días",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
))

# Líneas de referencia (umbral bajo/medio) usando paleta
fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_low, thr_low],
    mode="lines", line=dict(color=COLOR_MAP["Bajo"], dash="dot"),
    name=f"Bajo (≤ {thr_low:.3f})", hoverinfo="skip"))
fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_med, thr_med],
    mode="lines", line=dict(color=COLOR_MAP["Medio"], dash="dot"),
    name=f"Medio (≤ {thr_med:.3f})", hoverinfo="skip"))
fig_er.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
    line=dict(color=COLOR_MAP["Alto"], dash="dot"),
    name=f"Alto (> {thr_med:.3f})", hoverinfo="skip"))

fi, ff = x.min(), x.max()
fig_er.update_layout(
    xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
    hovermode="x unified", legend_title="Referencias", height=650
)
fig_er.update_xaxes(range=[fi, ff], dtick="D1" if (ff-fi).days <= 31 else "M1",
                    tickformat="%d-%b" if (ff-fi).days <= 31 else "%b")
fig_er.update_yaxes(rangemode="tozero")
st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

# ====== FIGURA: EMERGENCIA ACUMULADA ======
st.subheader("EMERGENCIA ACUMULADA DIARIA")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pred_vis["Fecha"], y=y_max, mode="lines", line=dict(width=0),
    name="Máximo", hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=pred_vis["Fecha"], y=y_min, mode="lines", line=dict(width=0),
    fill="tonexty", name="Mínimo",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
))
fig.add_trace(go.Scatter(
    x=pred_vis["Fecha"], y=y_adj, mode="lines", line=dict(width=2.5),
    name=f"Umbral ajustable (/{EMEAC_ADJ_DEN:.2f})",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"
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

# ====== TABLA: Resultados ======
st.subheader(f"Resultados ({rango_txt}) - {nombre}")
col_emeac = "EMEAC (%) - ajustable" if rango_opcion == "Todo el empalme" else "EMEAC (%) - ajustable (rango)"
nivel_icono = {"Bajo": "🟢 Bajo", "Medio": "🟡 Medio", "Alto": "🔴 Alto"}
tabla_rango = pred_vis[["Fecha","Julian_days","Nivel_Emergencia_relativa",col_emeac]].copy()
tabla_rango["Nivel_Emergencia_relativa"] = tabla_rango["Nivel_Emergencia_relativa"].map(nivel_icono)
tabla_rango = tabla_rango.rename(columns={"Nivel_Emergencia_relativa":"Nivel de EMERREL", col_emeac:"EMEAC (%)"})
tabla_rango = tabla_rango.sort_values("Fecha").reset_index(drop=True)
tabla_rango["Nivel de EMERREL"] = tabla_rango["Nivel de EMERREL"].fillna("🟢 Bajo")
tabla_rango["EMEAC (%)"] = pd.to_numeric(tabla_rango["EMEAC (%)"], errors="coerce").fillna(0).clip(0, 100)

st.dataframe(tabla_rango, use_container_width=True)

csv_buf = StringIO(); tabla_rango.to_csv(csv_buf, index=False)
st.download_button(
    f"Descargar resultados ({'todo' if rango_opcion=='Todo el empalme' else 'rango'}) - {nombre}",
    data=csv_buf.getvalue(),
    file_name=f"{nombre.replace(' ','_')}_{'todo' if rango_opcion=='Todo el empalme' else 'rango'}.csv",
    mime="text/csv"
)
