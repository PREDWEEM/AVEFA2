# AVEFA – Streamlit App Pack

Este paquete incluye:
- `app_emergencia.py` — App principal de AVEFA (con soporte para pronóstico completo, MA5 sombreada y autotest).
- `requirements.txt` — Dependencias mínimas.
- Usa pesos (`IW.npy`, `LW.npy`, `bias_IW.npy`, `bias_out.npy`) desde:
  `https://raw.githubusercontent.com/PREDWEEM/AVEFA2/main/`

## Cómo correr
```bash
pip install -r requirements.txt
streamlit run app_emergencia.py
```

## Secrets opcionales
En `.streamlit/secrets.toml` podés definir:
```toml
LOCAL_HISTORY_PATH = "avefa_history_local.csv"
FREEZE_HISTORY = false
```
