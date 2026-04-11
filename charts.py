"""
charts.py — Módulo de Visualizaciones y Gráficos
=================================================
Contiene todas las funciones que construyen y renderizan los gráficos
Plotly y Matplotlib del simulador. Cada función es autocontenida:
recibe los datos que necesita y llama a st.plotly_chart() / st.pyplot()
internamente.

Funciones:
    plot_cluster_bars()     → Gráfico de barras horizontales (carga % y kW por gen.)
    plot_cluster_kmeans()   → Gráfico de clúster estilo K-Means (nube de puntos)
    plot_convergence()      → Curva de convergencia del GA (costo limpio vs. generación)
    plot_fuzzy_membership() → Conjunto difuso defuzzificado de skfuzzy (Matplotlib)
    plot_dispatch_comparison() → GA vs. Greedy (barras dobles kW y costo por gen.)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


# ====================================================================
# GRÁFICO 1: BARRAS HORIZONTALES POR GENERADOR (% Carga y kW)
# ====================================================================
def plot_cluster_bars(best_chrom, allocation, GENERATORS, N_GENES, demand_val, final_prod):
    """
    Gráfico de barras horizontales con dos paneles (subplots):
      - Panel izquierdo:  % de carga asignado a cada generador
      - Panel derecho:    kW despachados por generador + línea de demanda FIS

    El color de cada barra codifica el perfil económico:
      Verde  → generador eficiente (≤$90/kW)
      Azul   → costo moderado ($91–$130/kW)
      Rojo   → BACKUP / EMERGENCIA (>$130/kW)
      Gris   → apagado

    Parámetros
    ----------
    best_chrom  : ndarray (8,) — genes óptimos del GA (% de carga)
    allocation  : ndarray (8,) — kW asignados por generador
    GENERATORS  : ndarray (8,2) — [capacidad, costo] por generador
    N_GENES     : int
    demand_val  : float — demanda en kW estimada por el FIS
    final_prod  : float — potencia total despachada por el GA
    """
    st.markdown("#### 📊 Gráfico del Clúster — Carga (%) y Potencia (kW) por Generador")

    # Vectores de datos
    gen_nombres      = [f"Gen {i+1}  ({int(GENERATORS[i,0])} kW máx)" for i in range(N_GENES)]
    cargas_pct       = [int(best_chrom[i]) for i in range(N_GENES)]
    kw_vals          = [float(allocation[i]) for i in range(N_GENES)]
    costos_unitarios = [float(GENERATORS[i, 1]) for i in range(N_GENES)]

    # Paleta de colores según costo unitario
    def _color(costo_unit, activo):
        if not activo:
            return 'rgba(80,80,90,0.5)'
        if costo_unit <= 90:
            return 'rgba(0,220,160,0.85)'
        elif costo_unit <= 130:
            return 'rgba(100,180,255,0.85)'
        return 'rgba(255,100,80,0.85)'

    colores = [_color(costos_unitarios[i], cargas_pct[i] > 0) for i in range(N_GENES)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Carga Asignada por el AG (%)", "Potencia Despachada (kW)"),
        horizontal_spacing=0.12
    )

    # Panel izquierdo: % de carga
    fig.add_trace(go.Bar(
        y=gen_nombres, x=cargas_pct, orientation='h', name='Carga (%)',
        marker=dict(color=colores, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f'{v}%' for v in cargas_pct],
        textposition='outside', textfont=dict(color='#eee', size=11),
        hovertemplate='<b>%{y}</b><br>Carga: %{x}%<extra></extra>',
    ), row=1, col=1)
    fig.add_vline(x=100, line_dash='dot', line_color='rgba(255,255,255,0.2)', row=1, col=1)

    # Panel derecho: kW despachados + línea de demanda
    fig.add_trace(go.Bar(
        y=gen_nombres, x=kw_vals, orientation='h', name='Potencia (kW)',
        marker=dict(color=colores, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f'{v:.0f} kW' for v in kw_vals],
        textposition='outside', textfont=dict(color='#eee', size=11),
        hovertemplate='<b>%{y}</b><br>Potencia: %{x} kW<extra></extra>',
    ), row=1, col=2)
    fig.add_vline(
        x=demand_val, line_dash='dash', line_color='#f2c94c',
        annotation_text=f"Demanda: {demand_val:.0f} kW",
        annotation_font_color='#f2c94c', annotation_position='top right',
        row=1, col=2
    )

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#161b26', paper_bgcolor='#0e1117',
        font=dict(color='#ddd'), height=360, showlegend=False,
        margin=dict(l=10, r=20, t=55, b=10),
        title=dict(
            text=f"Estado del Clúster Diésel · Demanda FIS: {demand_val:.0f} kW · Despacho GA: {final_prod:.0f} kW",
            x=0.5, xanchor='center', font=dict(size=13, color='#eee')
        ),
    )
    fig.update_xaxes(gridcolor='#2a2a3a', range=[0, 115], row=1, col=1)
    fig.update_xaxes(gridcolor='#2a2a3a', row=1, col=2)
    fig.update_yaxes(gridcolor='#2a2a3a')

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🟢 Verde: eficientes (≤$90/kW)  |  "
        "🔵 Azul: moderados ($91–$130/kW)  |  "
        "🔴 Rojo: BACKUP / EMERGENCIA (>$130/kW)  |  "
        "⬜ Gris: apagado"
    )


# ====================================================================
# GRÁFICO 2: CLÚSTER ESTILO K-MEANS (NUBE DE PUNTOS)
# ====================================================================
def plot_cluster_kmeans(best_chrom, allocation, GENERATORS, N_GENES, final_prod, final_cost, gap_pct):
    """
    Gráfico de dispersión estilo K-Means: muestra los 8 generadores
    agrupados en 3 clústeres por perfil económico (costo unitario), con
    nubes de puntos normalmente distribuidas alrededor de cada centroide.

    Clústeres:
      🟢 Eficientes ($70–$90/kW):  Gen 4, 5, 8
      🟡 Moderados  ($100–$110/kW): Gen 2, 7
      🔴 Costosos   ($150–$250/kW): Gen 1, 3, 6

    Las estrellas (⭐) marcan los generadores reales del sistema.
    El tamaño de la burbuja es fijo; la actividad se muestra por color.

    Parámetros
    ----------
    best_chrom : ndarray (8,) — genes óptimos del GA (% de carga)
    allocation : ndarray (8,) — kW asignados por generador
    GENERATORS : ndarray (8,2)
    N_GENES    : int
    final_prod : float — potencia total GA (kW)
    final_cost : float — costo GA (USD)
    gap_pct    : float — gap vs. Greedy (%)
    """
    st.markdown("#### 🔵 Gráfico de Clúster — Agrupamiento por Perfil Económico (Estilo K-Means)")
    st.caption(
        "Cada nube de puntos representa un clúster de generadores con costo operativo similar. "
        "La **estrella ⭐** indica el generador real activo; el **círculo abierto** indica apagado. "
        "El GA siempre prioriza el clúster 🟢 verde (más económico)."
    )

    # Semilla fija para que la nube no cambie en cada interacción
    rng_km = np.random.default_rng(seed=42)

    # Definición de los 3 clústeres: índices en GENERATORS (base-0)
    CLUSTERS_KM = [
        {"nombre": "Clúster 1 — Eficientes",  "indices": [3, 4, 7], "color": "#2ecc71", "rgba": "rgba(46,204,113,0.50)",  "n": 65},
        {"nombre": "Clúster 2 — Moderados",   "indices": [1, 6],    "color": "#f1c40f", "rgba": "rgba(241,196,15,0.50)", "n": 50},
        {"nombre": "Clúster 3 — Costosos",    "indices": [0, 2, 5], "color": "#e74c3c", "rgba": "rgba(231,76,60,0.50)",  "n": 65},
    ]

    fig = go.Figure()

    for c in CLUSTERS_KM:
        idx = c["indices"]
        # Centroide del clúster (media de costo y capacidad de sus generadores)
        cx = np.mean([GENERATORS[i, 1] for i in idx])
        cy = np.mean([GENERATORS[i, 0] for i in idx])
        # Dispersión proporcional al rango del clúster
        sx = np.std([GENERATORS[i, 1] for i in idx]) + 8
        sy = np.std([GENERATORS[i, 0] for i in idx]) + 55
        # Nube de puntos alrededor del centroide (distribución normal)
        cx_pts = rng_km.normal(cx, sx, c["n"])
        cy_pts = rng_km.normal(cy, sy, c["n"])

        # Traza 1: nube de puntos del clúster (sin hover, solo visual)
        fig.add_trace(go.Scatter(
            x=cx_pts, y=cy_pts, mode='markers', name=c["nombre"],
            marker=dict(size=7, color=c["rgba"], line=dict(color=c["color"], width=0.8)),
            hoverinfo='skip', showlegend=True,
        ))

        # Traza 2: generadores reales (estrella si activo, círculo abierto si apagado)
        for i in idx:
            activo = best_chrom[i] > 0
            fig.add_trace(go.Scatter(
                x=[GENERATORS[i, 1]], y=[GENERATORS[i, 0]],
                mode='markers+text', showlegend=False,
                marker=dict(
                    size=20 if activo else 13,
                    color=c["color"] if activo else 'rgba(100,100,110,0.6)',
                    symbol='star' if activo else 'circle-open',
                    line=dict(color='white' if activo else '#555', width=1.5),
                    opacity=1.0 if activo else 0.6,
                ),
                text=[f"<b>G{i+1}</b>"],
                textposition='top center',
                textfont=dict(color='white', size=10, family='Arial Black'),
                hovertext=(
                    f"<b>Gen {i+1}</b><br>"
                    f"Clúster: {c['nombre']}<br>"
                    f"Capacidad: {int(GENERATORS[i,0])} kW<br>"
                    f"Costo unitario: ${int(GENERATORS[i,1])}/kW<br>"
                    f"kW asignados: {float(allocation[i]):.1f}<br>"
                    f"Carga: {int(best_chrom[i])}%<br>"
                    f"Costo parcial: ${float(allocation[i])*GENERATORS[i,1]:,.0f}<br>"
                    f"Estado: {'⭐ OPERANDO' if activo else '⚫ APAGADO'}"
                ),
                hoverinfo='text',
            ))

    fig.update_layout(
        template='plotly_dark', plot_bgcolor='#0f1319', paper_bgcolor='#0e1117',
        font=dict(color='#ddd', family='Arial'), height=470,
        margin=dict(l=30, r=30, t=75, b=70),
        title=dict(
            text=(
                f"Clúster de Generadores (K-Means)  ·  "
                f"Despacho GA: <b>{final_prod:.0f} kW</b>  ·  "
                f"Costo: <b>${final_cost:,.0f}</b>  ·  Gap vs Greedy: <b>{gap_pct:.1f}%</b>"
            ),
            x=0.5, xanchor='center', font=dict(size=13, color='#eee')
        ),
        xaxis=dict(title='Costo Operativo Unitario (USD/kW)', gridcolor='#1a1f2e', zeroline=False, color='#888', tickprefix='$'),
        yaxis=dict(title='Capacidad Máxima (kW)', gridcolor='#1a1f2e', zeroline=False, color='#888'),
        legend=dict(orientation='h', y=-0.20, x=0.5, xanchor='center', font=dict(size=11, color='#ccc'), bgcolor='rgba(0,0,0,0)'),
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⭐ Estrella grande = generador activo  |  ○ Círculo = apagado por el AG  |  "
        "🟢 Eficientes ($70–$90/kW)  ·  🟡 Moderados ($100–$110/kW)  ·  🔴 Costosos ($150–$250/kW)"
    )


# ====================================================================
# GRÁFICO 3: CURVA DE CONVERGENCIA DEL GA
# ====================================================================
def plot_convergence(cost_history, gen_stopped, generations):
    """
    Gráfica de línea del costo operativo mínimo válido (sin penalización)
    a lo largo de las generaciones del GA. Los gaps NaN son generaciones
    donde ningún individuo satisfacía la restricción de demanda g(x).

    También marca con una línea vertical la primera generación factible
    (si tardó más de 2 generaciones en aparecer).

    Parámetros
    ----------
    cost_history : list[float] — costo mínimo limpio por generación (NaN si infactible)
    gen_stopped  : int         — generación en que el GA convergió/paró
    generations  : int         — límite máximo de generaciones configurado
    """
    actual_gens    = len(cost_history)
    gens_axis      = list(range(1, actual_gens + 1))
    valid_pairs    = [(g, c) for g, c in zip(gens_axis, cost_history) if not np.isnan(c)]
    first_feasible = valid_pairs[0][0] if valid_pairs else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gens_axis, y=cost_history,
        mode='lines', name='Costo Óptimo Válido (USD)',
        line=dict(color='#00ffcc', width=3),
        fill='tozeroy', fillcolor='rgba(0,255,204,0.07)',
        connectgaps=False,   # NaN → gaps = escapes del espacio infactible
    ))

    if first_feasible and first_feasible > 2:
        fig.add_vline(
            x=first_feasible, line_dash="dash", line_color="#f2c94c",
            annotation_text=f"Primera solución factible (gen {first_feasible})",
            annotation_font_color="#f2c94c", annotation_position="top right"
        )

    convergence_note = (
        f"Convergencia detectada en gen <b>{gen_stopped}</b> / {generations} "
        f"(PATIENCE={max(20, generations//5)})"
        if gen_stopped < generations else
        f"Ejecutó las {generations} generaciones completas"
    )

    fig.update_layout(
        title=dict(
            text=f"Convergencia GA — Costo Operativo Mínimo · {convergence_note}",
            x=0.5, xanchor='center', font=dict(color='#eee', size=13)
        ),
        xaxis=dict(title="Generación (t)", gridcolor='#2a2a3a', color='#ccc'),
        yaxis=dict(title="Costo USD", gridcolor='#2a2a3a', color='#ccc'),
        template="plotly_dark", plot_bgcolor="#161b26", paper_bgcolor="#0e1117",
        font=dict(color='#ddd'),
        legend=dict(orientation='h', y=1.1, font=dict(color='#ddd')),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "ℹ️ Los gaps iniciales (si existen) indican generaciones donde ningún individuo "
        "satisfacía g(x) (espacio infactible). Ilustra el escape de la barrera de penalización."
    )


# ====================================================================
# GRÁFICO 4: CONJUNTO DIFUSO DEFUZZIFICADO (skfuzzy / Matplotlib)
# ====================================================================
def plot_fuzzy_membership(demand_sim, demand_var, plot_fuzzy_result_fn):
    """
    Renderiza el gráfico nativo de skfuzzy con el conjunto de pertenencia
    del consecuente (Demanda) activado según la inferencia actual.

    Aplica un estilo oscuro consistente con el resto de la UI.
    Usa plt.close() para liberar memoria tras renderizar.

    Parámetros
    ----------
    demand_sim          : ControlSystemSimulation — objeto con la inferencia actual
    demand_var          : ctrl.Consequent         — variable consecuente del FIS
    plot_fuzzy_result_fn: callable — función plot_fuzzy_result de fuzzy_engine.py
    """
    st.markdown("**Conjunto Difuso Mamdani Defuzzificado (9 reglas AND):**")
    try:
        fig_f = plot_fuzzy_result_fn(demand_sim, demand_var)
        fig_f.patch.set_facecolor('#0e1117')
        ax = fig_f.gca()
        ax.set_facecolor('#0e1117')
        for spine in ax.spines.values():
            spine.set_color('#555')
        ax.xaxis.label.set_color('#ddd')
        ax.yaxis.label.set_color('#ddd')
        ax.tick_params(axis='x', colors='#ccc')
        ax.tick_params(axis='y', colors='#ccc')
        st.pyplot(fig_f)
        plt.close(fig_f)
    except Exception as e:
        st.warning(f"Render nativo de skfuzzy no disponible: {e}")


# ====================================================================
# GRÁFICO 5: DESPACHO ECONÓMICO GA vs. GREEDY (BARRAS DOBLES)
# ====================================================================
def plot_dispatch_comparison(allocation, greedy_alloc, GENERATORS, N_GENES,
                              final_prod, greedy_kw, final_cost, greedy_cost,
                              gap_pct, gap_emoji, best_chrom):
    """
    Gráfico de barras agrupadas (subplots) comparando la solución del GA
    con la solución Greedy generador por generador:
      - Subplot izquierdo: kW asignados por generador (GA vs. Greedy)
      - Subplot derecho:   Costo USD por generador   (GA vs. Greedy)

    También incluye una línea roja discontinua de capacidad máxima.

    Parámetros
    ----------
    allocation   : ndarray (8,) — kW asignados por el GA
    greedy_alloc : ndarray (8,) — kW asignados por el Greedy
    GENERATORS   : ndarray (8,2)
    N_GENES      : int
    final_prod   : float — potencia total GA
    greedy_kw    : float — potencia total Greedy
    final_cost   : float — costo total GA
    greedy_cost  : float — costo total Greedy
    gap_pct      : float — gap porcentual
    gap_emoji    : str   — emoji de color del gap
    best_chrom   : ndarray (8,) — cromosoma óptimo (para colorear barras GA)
    """
    st.markdown("### 📈 Despacho Económico Final — GA vs. Óptimo Greedy")
    st.caption(
        "El Greedy es la solución analítica **óptima** para funciones de costo lineales "
        "(Principio de Optimalidad de Bellman). Sirve como referencia para cuantificar la "
        "calidad del GA. Con modelos de costo no-lineales, el Greedy falla y el GA se justifica."
    )

    gen_labels      = [f"Gen {i+1}<br>({int(GENERATORS[i,0])}kW)" for i in range(N_GENES)]
    ga_costs_op     = [allocation[i] * GENERATORS[i, 1]   for i in range(N_GENES)]
    greedy_costs_op = [greedy_alloc[i] * GENERATORS[i, 1] for i in range(N_GENES)]

    # Colores GA: gradiente basado en % de carga del cromosoma
    bar_colors_ga = [
        f"rgba(0,{int(200*(best_chrom[i]/100))},{int(180*(best_chrom[i]/100))},0.85)"
        if best_chrom[i] > 0 else "rgba(50,50,60,0.5)"
        for i in range(N_GENES)
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Potencia Asignada por Gen. (kW)  |  GA={final_prod:.0f} kW  vs  Greedy={greedy_kw:.0f} kW",
            f"Costo por Gen. (USD)  |  GA=${final_cost:,.0f}  vs  Greedy=${greedy_cost:,.0f}",
        ),
        horizontal_spacing=0.1
    )

    # Subplot 1: kW asignados — GA, Greedy y línea de capacidad máxima
    fig.add_trace(go.Bar(
        x=gen_labels, y=list(allocation), name="GA (kW)",
        marker=dict(color=bar_colors_ga, line=dict(color='rgba(0,255,204,0.4)', width=1)),
        text=[f"<b>{v:.0f}</b>" for v in allocation],
        textposition='outside', textfont=dict(color='#00ffcc', size=10),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=gen_labels, y=list(greedy_alloc), name="Greedy (kW)",
        marker=dict(color='rgba(150,180,255,0.4)', line=dict(color='#6699ff', width=1)),
        text=[f"<b>{v:.0f}</b>" for v in greedy_alloc],
        textposition='outside', textfont=dict(color='#6699ff', size=10),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=gen_labels, y=list(GENERATORS[:, 0]),
        mode='lines+markers', name="Cap. máx.",
        line=dict(color='#ff6b6b', dash='dash', width=2),
        marker=dict(size=6, symbol='diamond'),
    ), row=1, col=1)

    # Subplot 2: Costo por generador — GA vs. Greedy
    fig.add_trace(go.Bar(
        x=gen_labels, y=ga_costs_op, name="Costo GA (USD)",
        marker=dict(color='#f2c94c', opacity=0.85),
        text=[f"<b>${v:,.0f}</b>" for v in ga_costs_op],
        textposition='outside', textfont=dict(color='#f2c94c', size=10),
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=gen_labels, y=greedy_costs_op, name="Costo Greedy (USD)",
        marker=dict(color='rgba(102,153,255,0.5)', line=dict(color='#6699ff', width=1)),
        text=[f"<b>${v:,.0f}</b>" for v in greedy_costs_op],
        textposition='outside', textfont=dict(color='#6699ff', size=10),
    ), row=1, col=2)

    fig.update_layout(
        barmode='group', template="plotly_dark",
        plot_bgcolor="#161b26", paper_bgcolor="#0e1117",
        font=dict(color='#ddd'), height=440, showlegend=True,
        legend=dict(orientation='h', y=1.14, x=0.25, font=dict(size=11)),
        title=dict(
            text=f"Cromosoma óptimo x* · Costo GA: <b>${final_cost:,.0f}</b> · "
                 f"Greedy: <b>${greedy_cost:,.0f}</b> · Gap: <b>{gap_pct:.1f}%</b> {gap_emoji}",
            x=0.5, xanchor='center', font=dict(size=13)
        ),
    )
    fig.update_yaxes(gridcolor='#2a2a3a', row=1, col=1)
    fig.update_yaxes(gridcolor='#2a2a3a', row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)
