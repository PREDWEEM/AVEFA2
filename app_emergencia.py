# ====== FIGURA: EMERGENCIA RELATIVA DIARIA (MA5 con relleno tricolor INTERNO, sin opacidad) ======
st.subheader("EMERGENCIA RELATIVA DIARIA")
fig_er = go.Figure()

# Barras coloreadas por nivel (usa COLOR_MAP)
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

# Área interna bajo MA5 segmentada: 0→0.01 (verde), 0.01→0.05 (amarillo), 0.05→MA5 (rojo)
x = pred_vis["Fecha"]
ma = pred_vis["EMERREL_MA5"].clip(lower=0)
thr_low = float(THR_BAJO_MEDIO)   # 0.01
thr_med = float(THR_MEDIO_ALTO)   # 0.05

# Topes por banda
y0 = np.zeros(len(ma))
y1 = np.minimum(ma, thr_low)   # tope verde
y2 = np.minimum(ma, thr_med)   # tope amarillo
y3 = ma                        # tope rojo

GREEN  = "#00A651"
YELLOW = "#FFC000"
RED    = "#E53935"

# 1) Base (0) — sin fill (ancora para la primera banda)
fig_er.add_trace(go.Scatter(
    x=x, y=y0, mode="lines",
    line=dict(width=0),
    hoverinfo="skip", showlegend=False
))

# 2) Banda VERDE: fill de y1 hacia y0
fig_er.add_trace(go.Scatter(
    x=x, y=y1, mode="lines",
    line=dict(width=0),
    fill="tonexty", fillcolor=GREEN,
    hoverinfo="skip", showlegend=False, name="Zona baja (verde)"
))

# 3) Baseline y1 (para que la siguiente banda llene contra y1)
fig_er.add_trace(go.Scatter(
    x=x, y=y1, mode="lines",
    line=dict(width=0),
    hoverinfo="skip", showlegend=False
))

# 4) Banda AMARILLA: fill de y2 hacia y1
fig_er.add_trace(go.Scatter(
    x=x, y=y2, mode="lines",
    line=dict(width=0),
    fill="tonexty", fillcolor=YELLOW,
    hoverinfo="skip", showlegend=False, name="Zona media (amarillo)"
))

# 5) Baseline y2 (para que la siguiente banda llene contra y2)
fig_er.add_trace(go.Scatter(
    x=x, y=y2, mode="lines",
    line=dict(width=0),
    hoverinfo="skip", showlegend=False
))

# 6) Banda ROJA: fill de y3 (MA5) hacia y2
fig_er.add_trace(go.Scatter(
    x=x, y=y3, mode="lines",
    line=dict(width=0),
    fill="tonexty", fillcolor=RED,
    hoverinfo="skip", showlegend=False, name="Zona alta (rojo)"
))

# Línea MA5 por encima del relleno
fig_er.add_trace(go.Scatter(
    x=x, y=ma, mode="lines",
    line=dict(width=2),
    name="Media móvil 5 días",
    hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
))

# Líneas de referencia (umbral bajo y medio) usando la paleta global
fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_low, thr_low],
    mode="lines", line=dict(color=COLOR_MAP["Bajo"], dash="dot"),
    name=f"Bajo (≤ {thr_low:.3f})", hoverinfo="skip"))
fig_er.add_trace(go.Scatter(x=[x.min(), x.max()], y=[thr_med, thr_med],
    mode="lines", line=dict(color=COLOR_MAP["Medio"], dash="dot"),
    name=f"Medio (≤ {thr_med:.3f})", hoverinfo="skip"))
fig_er.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
    line=dict(color=COLOR_MAP["Alto"], dash="dot"),
    name=f"Alto (> {thr_med:.3f})", hoverinfo="skip"))

# Layout
fi, ff = x.min(), x.max()
fig_er.update_layout(
    xaxis_title="Fecha", yaxis_title="EMERREL (0-1)",
    hovermode="x unified", legend_title="Referencias", height=650
)
fig_er.update_xaxes(range=[fi, ff], dtick="D1" if (ff-fi).days <= 31 else "M1",
                    tickformat="%d-%b" if (ff-fi).days <= 31 else "%b")
fig_er.update_yaxes(rangemode="tozero")

st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")
