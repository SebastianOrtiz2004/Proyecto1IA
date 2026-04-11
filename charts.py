"""
charts.py — Módulo de Visualizaciones y Gráficos
=================================================
Contiene todas las funciones que construyen y renderizan los gráficos
Plotly y Matplotlib del simulador. Cada función es autocontenida:
recibe los datos que necesita y llama a st.plotly_chart() / st.pyplot()
internamente.

Funciones:
    graficar_cluster_barras()      → Barras horizontales (% carga y kW por generador)
    graficar_cluster_kmeans()      → Nube de puntos estilo K-Medias
    graficar_convergencia()        → Curva de costo mínimo vs. generación del AG
    graficar_membresia_difusa()    → Conjunto difuso defuzzificado (skfuzzy)
    graficar_comparacion_despacho()→ AG vs. Voraz (barras dobles kW y costo)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


# ====================================================================
# GRÁFICO 1: BARRAS HORIZONTALES POR GENERADOR (% Carga y kW)
# ====================================================================
def graficar_cluster_barras(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES,
                             demanda_kw, potencia_ag):
    """
    Gráfico de barras horizontales con dos paneles:
      - Panel izquierdo:  % de carga asignado a cada generador
      - Panel derecho:    kW despachados + línea de demanda del SID

    Paleta de colores según perfil económico:
      Verde  → eficiente (≤$90/kW)
      Azul   → moderado ($91–$130/kW)
      Rojo   → RESPALDO / EMERGENCIA (>$130/kW)
      Gris   → apagado por el AG

    Parámetros
    ----------
    mejor_cromosoma : ndarray (8,) — genes óptimos del AG (% de carga)
    asignacion      : ndarray (8,) — kW asignados por generador
    GENERADORES     : ndarray (8,2)
    N_GENERADORES   : int
    demanda_kw      : float — demanda estimada por el SID
    potencia_ag     : float — potencia total despachada por el AG
    """
    st.markdown("#### 📊 Gráfico del Clúster — Carga (%) y Potencia (kW) por Generador")

    # Vectores de datos
    nombres_gen      = [f"Gen {i+1}  ({int(GENERADORES[i,0])} kW máx)" for i in range(N_GENERADORES)]
    porcentajes      = [int(mejor_cromosoma[i]) for i in range(N_GENERADORES)]
    valores_kw       = [float(asignacion[i]) for i in range(N_GENERADORES)]
    costos_unitarios = [float(GENERADORES[i, 1]) for i in range(N_GENERADORES)]

    # Función local de color según costo unitario y estado
    def _color_barra(costo_unit, activo):
        if not activo:
            return 'rgba(80,80,90,0.5)'
        if costo_unit <= 90:
            return 'rgba(0,220,160,0.85)'
        elif costo_unit <= 130:
            return 'rgba(100,180,255,0.85)'
        return 'rgba(255,100,80,0.85)'

    colores = [_color_barra(costos_unitarios[i], porcentajes[i] > 0) for i in range(N_GENERADORES)]

    figura = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Carga Asignada por el AG (%)", "Potencia Despachada (kW)"),
        horizontal_spacing=0.12
    )

    # Panel izquierdo: % de carga por generador
    figura.add_trace(go.Bar(
        y=nombres_gen, x=porcentajes, orientation='h', name='Carga (%)',
        marker=dict(color=colores, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f'{v}%' for v in porcentajes],
        textposition='outside', textfont=dict(color='#eee', size=11),
        hovertemplate='<b>%{y}</b><br>Carga: %{x}%<extra></extra>',
    ), row=1, col=1)
    figura.add_vline(x=100, line_dash='dot', line_color='rgba(255,255,255,0.2)', row=1, col=1)

    # Panel derecho: kW despachados + línea de demanda
    figura.add_trace(go.Bar(
        y=nombres_gen, x=valores_kw, orientation='h', name='Potencia (kW)',
        marker=dict(color=colores, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f'{v:.0f} kW' for v in valores_kw],
        textposition='outside', textfont=dict(color='#eee', size=11),
        hovertemplate='<b>%{y}</b><br>Potencia: %{x} kW<extra></extra>',
    ), row=1, col=2)
    figura.add_vline(
        x=demanda_kw, line_dash='dash', line_color='#f2c94c',
        annotation_text=f"Demanda: {demanda_kw:.0f} kW",
        annotation_font_color='#f2c94c', annotation_position='top right',
        row=1, col=2
    )

    figura.update_layout(
        template='plotly_dark', plot_bgcolor='#161b26', paper_bgcolor='#0e1117',
        font=dict(color='#ddd'), height=360, showlegend=False,
        margin=dict(l=10, r=20, t=55, b=10),
        title=dict(
            text=f"Clúster Diésel · Demanda SID: {demanda_kw:.0f} kW · Despacho AG: {potencia_ag:.0f} kW",
            x=0.5, xanchor='center', font=dict(size=13, color='#eee')
        ),
    )
    figura.update_xaxes(gridcolor='#2a2a3a', range=[0, 115], row=1, col=1)
    figura.update_xaxes(gridcolor='#2a2a3a', row=1, col=2)
    figura.update_yaxes(gridcolor='#2a2a3a')

    st.plotly_chart(figura, use_container_width=True)
    st.caption(
        "🟢 Verde: eficientes (≤$90/kW)  |  "
        "🔵 Azul: moderados ($91–$130/kW)  |  "
        "🔴 Rojo: RESPALDO / EMERGENCIA (>$130/kW)  |  "
        "⬜ Gris: apagado por el optimizador"
    )


# ====================================================================
# GRÁFICO 2: CLÚSTER ESTILO K-MEDIAS (NUBE DE PUNTOS)
# ====================================================================
def graficar_cluster_kmeans(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES,
                             potencia_ag, costo_ag, brecha_pct):
    """
    Gráfico de dispersión estilo K-Medias: muestra los 8 generadores
    agrupados en 3 clústeres por perfil económico (costo unitario).
    Genera nubes de puntos normalmente distribuidas alrededor de cada centroide.

    Clústeres:
      🟢 Eficientes ($70–$90/kW):  Gen 4, 5, 8
      🟡 Moderados  ($100–$110/kW): Gen 2, 7
      🔴 Costosos   ($150–$250/kW): Gen 1, 3, 6

    Las estrellas (⭐) marcan los generadores activos del sistema.

    Parámetros
    ----------
    mejor_cromosoma : ndarray (8,)
    asignacion      : ndarray (8,)
    GENERADORES     : ndarray (8,2)
    N_GENERADORES   : int
    potencia_ag     : float — potencia total AG (kW)
    costo_ag        : float — costo AG (USD)
    brecha_pct      : float — brecha vs. Voraz (%)
    """
    st.markdown("#### 🔵 Gráfico de Clúster — Agrupamiento por Perfil Económico (Estilo K-Medias)")
    st.caption(
        "Cada nube de puntos representa un clúster de generadores con costo operativo similar. "
        "La **estrella ⭐** indica el generador activo; el **círculo abierto** indica apagado. "
        "El AG prioriza siempre el clúster 🟢 verde (más económico)."
    )

    # Semilla fija: la nube no cambia entre ejecuciones del simulador
    generador_aleatorio = np.random.default_rng(seed=42)

    # Definición de los 3 clústeres económicos (índices en GENERADORES base-0)
    CLUSTERS = [
        {"nombre": "Clúster 1 — Eficientes",  "indices": [3, 4, 7], "color": "#2ecc71", "rgba": "rgba(46,204,113,0.50)",  "num_puntos": 65},
        {"nombre": "Clúster 2 — Moderados",   "indices": [1, 6],    "color": "#f1c40f", "rgba": "rgba(241,196,15,0.50)", "num_puntos": 50},
        {"nombre": "Clúster 3 — Costosos",    "indices": [0, 2, 5], "color": "#e74c3c", "rgba": "rgba(231,76,60,0.50)",  "num_puntos": 65},
    ]

    figura = go.Figure()

    for cluster in CLUSTERS:
        indices = cluster["indices"]

        # Centroide del clúster: media de costo (eje X) y capacidad (eje Y)
        centro_x = np.mean([GENERADORES[i, 1] for i in indices])
        centro_y = np.mean([GENERADORES[i, 0] for i in indices])

        # Dispersión proporcional al rango del clúster más un margen base
        dispersion_x = np.std([GENERADORES[i, 1] for i in indices]) + 8
        dispersion_y = np.std([GENERADORES[i, 0] for i in indices]) + 55

        # Nube de puntos normalmente distribuida alrededor del centroide
        puntos_x = generador_aleatorio.normal(centro_x, dispersion_x, cluster["num_puntos"])
        puntos_y = generador_aleatorio.normal(centro_y, dispersion_y, cluster["num_puntos"])

        # Traza 1: nube de puntos del clúster (solo visual, sin hover)
        figura.add_trace(go.Scatter(
            x=puntos_x, y=puntos_y, mode='markers', name=cluster["nombre"],
            marker=dict(size=7, color=cluster["rgba"], line=dict(color=cluster["color"], width=0.8)),
            hoverinfo='skip', showlegend=True,
        ))

        # Traza 2: generadores reales (estrella si activo, círculo abierto si apagado)
        for i in indices:
            activo = mejor_cromosoma[i] > 0
            figura.add_trace(go.Scatter(
                x=[GENERADORES[i, 1]], y=[GENERADORES[i, 0]],
                mode='markers+text', showlegend=False,
                marker=dict(
                    size=20 if activo else 13,
                    color=cluster["color"] if activo else 'rgba(100,100,110,0.6)',
                    symbol='star' if activo else 'circle-open',
                    line=dict(color='white' if activo else '#555', width=1.5),
                    opacity=1.0 if activo else 0.6,
                ),
                text=[f"<b>G{i+1}</b>"],
                textposition='top center',
                textfont=dict(color='white', size=10, family='Arial Black'),
                hovertext=(
                    f"<b>Gen {i+1}</b><br>"
                    f"Clúster: {cluster['nombre']}<br>"
                    f"Capacidad: {int(GENERADORES[i,0])} kW<br>"
                    f"Costo unitario: ${int(GENERADORES[i,1])}/kW<br>"
                    f"kW asignados: {float(asignacion[i]):.1f}<br>"
                    f"Carga: {int(mejor_cromosoma[i])}%<br>"
                    f"Costo parcial: ${float(asignacion[i])*GENERADORES[i,1]:,.0f}<br>"
                    f"Estado: {'⭐ OPERANDO' if activo else '⚫ APAGADO'}"
                ),
                hoverinfo='text',
            ))

    figura.update_layout(
        template='plotly_dark', plot_bgcolor='#0f1319', paper_bgcolor='#0e1117',
        font=dict(color='#ddd', family='Arial'), height=470,
        margin=dict(l=30, r=30, t=75, b=70),
        title=dict(
            text=(
                f"Clúster de Generadores (K-Medias)  ·  "
                f"Despacho AG: <b>{potencia_ag:.0f} kW</b>  ·  "
                f"Costo: <b>${costo_ag:,.0f}</b>  ·  Brecha vs Voraz: <b>{brecha_pct:.1f}%</b>"
            ),
            x=0.5, xanchor='center', font=dict(size=13, color='#eee')
        ),
        xaxis=dict(title='Costo Operativo Unitario (USD/kW)', gridcolor='#1a1f2e',
                   zeroline=False, color='#888', tickprefix='$'),
        yaxis=dict(title='Capacidad Máxima (kW)', gridcolor='#1a1f2e', zeroline=False, color='#888'),
        legend=dict(orientation='h', y=-0.20, x=0.5, xanchor='center',
                    font=dict(size=11, color='#ccc'), bgcolor='rgba(0,0,0,0)'),
    )

    st.plotly_chart(figura, use_container_width=True)
    st.caption(
        "⭐ Estrella grande = generador activo  |  ○ Círculo = apagado por el AG  |  "
        "🟢 Eficientes ($70–$90/kW)  ·  🟡 Moderados ($100–$110/kW)  ·  🔴 Costosos ($150–$250/kW)"
    )


# ====================================================================
# GRÁFICO 3: CURVA DE CONVERGENCIA DEL AG
# ====================================================================
def graficar_convergencia(historial_costo, gen_parada, num_generaciones):
    """
    Gráfica de línea del costo operativo mínimo válido (sin penalización)
    a lo largo de las generaciones del AG.

    Los valores NaN corresponden a generaciones donde ningún cromosoma
    satisfacía la restricción de demanda g(x). Se representan como gaps
    en la curva, ilustrando el escape del espacio infactible.

    Parámetros
    ----------
    historial_costo  : list[float] — costo mínimo limpio por generación (NaN si infactible)
    gen_parada       : int         — generación en que el AG convergió
    num_generaciones : int         — límite máximo configurado
    """
    total_generaciones = len(historial_costo)
    eje_generaciones   = list(range(1, total_generaciones + 1))
    pares_validos      = [(g, c) for g, c in zip(eje_generaciones, historial_costo) if not np.isnan(c)]
    primera_factible   = pares_validos[0][0] if pares_validos else None

    figura = go.Figure()
    figura.add_trace(go.Scatter(
        x=eje_generaciones, y=historial_costo,
        mode='lines', name='Costo Óptimo Válido (USD)',
        line=dict(color='#00ffcc', width=3),
        fill='tozeroy', fillcolor='rgba(0,255,204,0.07)',
        connectgaps=False,   # NaN → gaps naturales en la curva
    ))

    if primera_factible and primera_factible > 2:
        figura.add_vline(
            x=primera_factible, line_dash="dash", line_color="#f2c94c",
            annotation_text=f"Primera sol. factible (gen {primera_factible})",
            annotation_font_color="#f2c94c", annotation_position="top right"
        )

    nota_convergencia = (
        f"Convergencia en gen <b>{gen_parada}</b> / {num_generaciones} "
        f"(PACIENCIA={max(20, num_generaciones//5)})"
        if gen_parada < num_generaciones else
        f"Ejecutó las {num_generaciones} generaciones completas"
    )

    figura.update_layout(
        title=dict(
            text=f"Convergencia AG — Costo Mínimo · {nota_convergencia}",
            x=0.5, xanchor='center', font=dict(color='#eee', size=13)
        ),
        xaxis=dict(title="Generación (t)", gridcolor='#2a2a3a', color='#ccc'),
        yaxis=dict(title="Costo USD", gridcolor='#2a2a3a', color='#ccc'),
        template="plotly_dark", plot_bgcolor="#161b26", paper_bgcolor="#0e1117",
        font=dict(color='#ddd'),
        legend=dict(orientation='h', y=1.1, font=dict(color='#ddd')),
    )
    st.plotly_chart(figura, use_container_width=True)
    st.caption(
        "ℹ️ Los gaps iniciales indican generaciones donde ningún cromosoma "
        "satisfacía g(x). Ilustra el escape de la barrera de penalización."
    )


# ====================================================================
# GRÁFICO 4: CONJUNTO DIFUSO DEFUZZIFICADO (skfuzzy / Matplotlib)
# ====================================================================
def graficar_membresia_difusa(simulacion_difusa, var_demanda, funcion_graficar):
    """
    Renderiza el gráfico nativo de skfuzzy con el conjunto de pertenencia
    del consecuente (Demanda) activado según la inferencia actual.

    Aplica estilo oscuro consistente con el resto de la interfaz.
    Llama plt.close() al final para liberar memoria.

    Parámetros
    ----------
    simulacion_difusa : ControlSystemSimulation — objeto con la inferencia actual
    var_demanda       : ctrl.Consequent         — variable consecuente del SID
    funcion_graficar  : callable — graficar_resultado_difuso de fuzzy_engine.py
    """
    st.markdown("**Conjunto Difuso Mamdani Defuzzificado (9 reglas AND):**")
    try:
        figura_difusa = funcion_graficar(simulacion_difusa, var_demanda)
        figura_difusa.patch.set_facecolor('#0e1117')
        eje = figura_difusa.gca()
        eje.set_facecolor('#0e1117')
        for borde in eje.spines.values():
            borde.set_color('#555')
        eje.xaxis.label.set_color('#ddd')
        eje.yaxis.label.set_color('#ddd')
        eje.tick_params(axis='x', colors='#ccc')
        eje.tick_params(axis='y', colors='#ccc')
        st.pyplot(figura_difusa)
        plt.close(figura_difusa)
    except Exception as error:
        st.warning(f"Renderizado del conjunto difuso no disponible: {error}")


# ====================================================================
# GRÁFICO 5: DESPACHO ECONÓMICO AG vs. VORAZ (BARRAS DOBLES)
# ====================================================================
def graficar_comparacion_despacho(asignacion, asignacion_voraz, GENERADORES, N_GENERADORES,
                                   potencia_ag, potencia_voraz, costo_ag, costo_voraz,
                                   brecha_pct, emoji_brecha, mejor_cromosoma):
    """
    Gráfico de barras agrupadas comparando AG vs. Voraz generador por generador:
      - Panel izquierdo: kW asignados (AG vs. Voraz + línea de capacidad máxima)
      - Panel derecho:   Costo USD por generador (AG vs. Voraz)

    Parámetros
    ----------
    asignacion       : ndarray (8,) — kW asignados por el AG
    asignacion_voraz : ndarray (8,) — kW asignados por el Voraz
    GENERADORES      : ndarray (8,2)
    N_GENERADORES    : int
    potencia_ag      : float
    potencia_voraz   : float
    costo_ag         : float
    costo_voraz      : float
    brecha_pct       : float
    emoji_brecha     : str
    mejor_cromosoma  : ndarray (8,)
    """
    st.markdown("### 📈 Despacho Económico Final — AG vs. Algoritmo Voraz")
    st.caption(
        "El Voraz es la referencia heurística óptima para costos lineales. "
        "Con modelo cuadrático-térmico (T > 0), el Voraz da solución subóptima "
        "porque ignora el costo marginal creciente con la carga. El AG puede superarlo."
    )

    etiquetas_gen  = [f"Gen {i+1}<br>({int(GENERADORES[i,0])}kW)" for i in range(N_GENERADORES)]
    costos_ag_op   = [asignacion[i] * GENERADORES[i, 1]       for i in range(N_GENERADORES)]
    costos_voraz_op = [asignacion_voraz[i] * GENERADORES[i, 1] for i in range(N_GENERADORES)]

    # Gradiente de color para barras del AG según % de carga
    colores_ag = [
        f"rgba(0,{int(200*(mejor_cromosoma[i]/100))},{int(180*(mejor_cromosoma[i]/100))},0.85)"
        if mejor_cromosoma[i] > 0 else "rgba(50,50,60,0.5)"
        for i in range(N_GENERADORES)
    ]

    figura = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"kW Asignados por Gen.  |  AG={potencia_ag:.0f} kW  vs  Voraz={potencia_voraz:.0f} kW",
            f"Costo por Gen. (USD)   |  AG=${costo_ag:,.0f}  vs  Voraz=${costo_voraz:,.0f}",
        ),
        horizontal_spacing=0.1
    )

    # Panel izquierdo: kW — AG, Voraz y línea de capacidad máxima
    figura.add_trace(go.Bar(
        x=etiquetas_gen, y=list(asignacion), name="AG (kW)",
        marker=dict(color=colores_ag, line=dict(color='rgba(0,255,204,0.4)', width=1)),
        text=[f"<b>{v:.0f}</b>" for v in asignacion],
        textposition='outside', textfont=dict(color='#00ffcc', size=10),
    ), row=1, col=1)

    figura.add_trace(go.Bar(
        x=etiquetas_gen, y=list(asignacion_voraz), name="Voraz (kW)",
        marker=dict(color='rgba(150,180,255,0.4)', line=dict(color='#6699ff', width=1)),
        text=[f"<b>{v:.0f}</b>" for v in asignacion_voraz],
        textposition='outside', textfont=dict(color='#6699ff', size=10),
    ), row=1, col=1)

    figura.add_trace(go.Scatter(
        x=etiquetas_gen, y=list(GENERADORES[:, 0]),
        mode='lines+markers', name="Cap. máx.",
        line=dict(color='#ff6b6b', dash='dash', width=2),
        marker=dict(size=6, symbol='diamond'),
    ), row=1, col=1)

    # Panel derecho: costo por generador — AG vs. Voraz
    figura.add_trace(go.Bar(
        x=etiquetas_gen, y=costos_ag_op, name="Costo AG (USD)",
        marker=dict(color='#f2c94c', opacity=0.85),
        text=[f"<b>${v:,.0f}</b>" for v in costos_ag_op],
        textposition='outside', textfont=dict(color='#f2c94c', size=10),
    ), row=1, col=2)

    figura.add_trace(go.Bar(
        x=etiquetas_gen, y=costos_voraz_op, name="Costo Voraz (USD)",
        marker=dict(color='rgba(102,153,255,0.5)', line=dict(color='#6699ff', width=1)),
        text=[f"<b>${v:,.0f}</b>" for v in costos_voraz_op],
        textposition='outside', textfont=dict(color='#6699ff', size=10),
    ), row=1, col=2)

    figura.update_layout(
        barmode='group', template="plotly_dark",
        plot_bgcolor="#161b26", paper_bgcolor="#0e1117",
        font=dict(color='#ddd'), height=440, showlegend=True,
        legend=dict(orientation='h', y=1.14, x=0.25, font=dict(size=11)),
        title=dict(
            text=f"Cromosoma óptimo x* · Costo AG: <b>${costo_ag:,.0f}</b> · "
                 f"Voraz: <b>${costo_voraz:,.0f}</b> · Brecha: <b>{brecha_pct:.1f}%</b> {emoji_brecha}",
            x=0.5, xanchor='center', font=dict(size=13)
        ),
    )
    figura.update_yaxes(gridcolor='#2a2a3a', row=1, col=1)
    figura.update_yaxes(gridcolor='#2a2a3a', row=1, col=2)
    st.plotly_chart(figura, use_container_width=True)
