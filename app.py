import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# ── Importaciones modulares (Arquitectura Plana) ──────────────────────────────
from fuzzy_engine import build_fuzzy_system, estimate_demand, plot_fuzzy_result
from genetic_optimizer import (
    run_genetic_algorithm, greedy_dispatch, GENERATORS, N_GENES
)

# ====================================================================
# 1. CONFIGURACIÓN DEL FRAMEWORK & CACHÉ DEL FIS
# ====================================================================
st.set_page_config(
    page_title="Simulador Planta Diésel — 8 Generadores",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_fuzzy_system():
    """
    El FIS Mamdani se compila UNA sola vez y se almacena en caché global de Streamlit.
    @st.cache_resource garantiza que el objeto ctrl.ControlSystem (grafo ponderado de
    reglas) no se reconstruya en cada interacción.  Reduce latencia ~500ms → ~5ms.
    """
    return build_fuzzy_system()


# ── CSS Premium (Glassmorphism + Animaciones Neón) ────────────────────────────
st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }

.kpi-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
    border-left: 5px solid #00ffcc; padding: 20px; border-radius: 8px;
    margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.3s;
}
.kpi-card:hover { transform: translateY(-5px); }

.gen-card {
    background: linear-gradient(145deg,rgba(20,25,35,0.9),rgba(15,20,30,0.9));
    border:1px solid rgba(255,255,255,0.05); border-radius:16px;
    padding:20px 15px; margin-bottom:18px; text-align:center;
    transition:all 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
    position:relative; overflow:hidden;
}
.gen-active  { border:1px solid #00ffcc; box-shadow:0 0 25px rgba(0,255,204,0.2),inset 0 0 20px rgba(0,255,204,0.05); }
.gen-inactive{ border:1px solid #ff3366; opacity:0.60; filter:grayscale(80%); }

.gen-icon { font-size:3rem; margin-bottom:12px; }
.active-icon  { text-shadow:0 0 15px #00ffcc,0 0 30px #00ffcc; animation:gen_pulse 1.5s infinite; }
.inactive-icon{ color:#444; }

@keyframes gen_pulse {
    0%  { transform:scale(1);    filter:brightness(1); }
    50% { transform:scale(1.08); filter:brightness(1.3); }
    100%{ transform:scale(1);    filter:brightness(1); }
}

.progress-rail {
    background:#1a1d24; border-radius:10px; height:14px; width:100%;
    margin-top:15px; box-shadow:inset 0 2px 4px rgba(0,0,0,0.5);
    position:relative; overflow:hidden;
}
.progress-fill {
    height:100%; background:linear-gradient(90deg,#00C9FF 0%,#00ffcc 100%);
    border-radius:10px; transition:width 1s cubic-bezier(0.22,1,0.36,1);
    box-shadow:0 0 10px #00ffcc;
}
.progress-fill-inactive { background:#444;height:100%;border-radius:10px; }

.gen-title  { font-size:1.2rem; font-weight:800; letter-spacing:1px; color:#fff; }
.gen-badge  { display:inline-block;padding:3px 9px;border-radius:12px;font-size:0.75rem;font-weight:bold;margin:8px 0; }
.badge-on   { background:rgba(0,255,204,0.2); color:#00ffcc; }
.badge-off  { background:rgba(255,51,102,0.2); color:#ff3366; }
.gen-data   { font-size:0.95rem; color:#ddd; margin:4px 0; }
.cost-data  { font-size:1rem; font-weight:bold; color:#f2c94c; }
.gen-footer { font-size:0.7rem;color:#888;margin-top:12px;border-top:1px dashed #333;padding-top:8px; }

.benchmark-box {
    background:rgba(242,201,76,0.08); border:1px solid rgba(242,201,76,0.3);
    border-radius:10px; padding:15px 20px; margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ── Títulos Principales ───────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;margin-bottom:0;'>⚡ Simulador de Despacho Económico — Planta Diésel 8 GEN</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:#aaa;margin-bottom:30px;'>"
    "Optimización Interactiva · Algoritmo Genético (NumPy) + Lógica Difusa Mamdani (skfuzzy) "
    "· Capacidad total instalada: <b>2950 kW</b></p>",
    unsafe_allow_html=True
)

# ====================================================================
# 2. SIDEBAR — Controles de Simulación
# ====================================================================
st.sidebar.markdown("### 🎛️ Variables de Entrada (Antecedentes FIS)")
st.sidebar.caption("Cada cambio re-ejecuta el FIS y el GA instantáneamente.")

temperature_val = st.sidebar.slider("🌡 Temperatura Externa (°C)", 0.0, 100.0, 60.0, step=1.0)
production_val  = st.sidebar.slider("🏭 Carga Productiva (%)",      0.0, 100.0, 75.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Parámetros del Algoritmo Genético")
st.sidebar.caption("Metaheurística de optimización combinatoria.")

pop_size      = st.sidebar.slider("Tamaño de Población (N)",   10, 200, 60)
generations   = st.sidebar.slider("Generaciones máx. (t_max)", 10, 300, 120)
mutation_rate = st.sidebar.slider("Tasa de Mutación (μ)",     0.01, 0.50, 0.10, step=0.01)
elite_k       = st.sidebar.slider("Élites preservados (k)",     1,   5,   2)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Espacio de búsqueda:** `|S| = 101⁸ ≈ 1.08×10¹⁶`  \n"
    f"**Paciencia (PATIENCE):** `{max(20, generations//5)}` gen. sin mejora"
)
st.sidebar.caption(
    "La magnitud del espacio justifica el uso de metaheurísticas en lugar de "
    "fuerza bruta o enumeración exhaustiva."
)

# ====================================================================
# FASE LÓGICA — FIS + GA + Greedy Benchmark
# ====================================================================
# ── Paso 1: Ejecutar el Sistema de Inferencia Difusa (FIS Mamdani) ──────────────
# El FIS toma los valores crisp (Temperatura y Producción) de los sliders,
# los fuzzifica usando funciones de pertenencia triangulares, aplica las
# 9 reglas AND de la base de conocimiento, y devuelve la demanda estimada
# en kW mediante el método del centroide (defuzzificación).
# demand_ctrl y demand_var están cacheados (@st.cache_resource): el grafo
# de reglas se compila UNA sola vez por sesión, no en cada interacción.
demand_ctrl, demand_var = get_fuzzy_system()
demand_val, demand_sim  = estimate_demand(demand_ctrl, demand_var, temperature_val, production_val)

# ── Paso 2: Ejecutar el Algoritmo Genético (meta-heurística de optimización) ────
# El GA recibe la demanda calculada por el FIS como restricción dura (Hard Constraint).
# Genera una población inicial aleatoria de N cromosomas, cada uno con 8 genes (0–100%).
# En cada generación aplica: Selección por torneo (k=3), Cruce de un punto (Pc=0.85),
# Mutación uniforme (μ=tasa configurada) y Elitismo (preserva los k mejores).
# Devuelve el cromosoma óptimo, el historial de fitness y la generación de convergencia.
best_chrom, allocation, final_cost, final_prod, fitness_history, cost_history, gen_stopped = \
    run_genetic_algorithm(
        demand=demand_val,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        elite_k=elite_k,
    )

# ── Paso 3: Calcular el Benchmark Greedy (solución analítica óptima) ────────────
# El Greedy ordena los generadores por costo unitario ascendente y asigna carga
# al más barato primero hasta cubrir la demanda. Para costos LINEALES separables,
# esta política es el ÓPTIMO GLOBAL garantizado (Principio de Optimalidad de Bellman).
# Se usa como referencia para medir la calidad de la solución del GA (Gap %).
greedy_alloc, greedy_cost, greedy_kw, greedy_pct = greedy_dispatch(demand_val)

# ====================================================================
# 3. KPI CARDS — Dashboard Central
# ====================================================================
col_a, col_b, col_c, col_d = st.columns(4)

col_a.markdown(f"""
<div class='kpi-card'>
    <div style='color:#00ffcc;'>🧠 <b>Demanda Estimada (FIS Mamdani)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>{demand_val:.1f} kW</div>
    <div style='font-size:0.8rem;color:#888;'>Centroide defuzzificado · 9 reglas AND</div>
</div>""", unsafe_allow_html=True)

delta = final_prod - demand_val
color_b = "#00ffcc" if final_prod >= demand_val else "#ff3366"
status  = f"Satisfecha ✅ (+{delta:.0f} kW)" if delta >= 0 else f"Déficit ❌ ({delta:.0f} kW)"

col_b.markdown(f"""
<div class='kpi-card' style='border-left-color:{color_b};'>
    <div style='color:{color_b};'>⚙️ <b>Potencia GA (cromosoma óptimo)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>{final_prod:.1f} kW</div>
    <div style='font-size:0.85rem;color:#ccc;'>{status}</div>
</div>""", unsafe_allow_html=True)

col_c.markdown(f"""
<div class='kpi-card' style='border-left-color:#f2c94c;'>
    <div style='color:#f2c94c;'>💸 <b>Costo GA (USD/h)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>${final_cost:,.0f}</div>
    <div style='font-size:0.8rem;color:#888;'>Sin penalización</div>
</div>""", unsafe_allow_html=True)

# ── Benchmark Greedy ─────────────────────────────────────────────────────────
gap_pct = ((final_cost - greedy_cost) / greedy_cost * 100) if greedy_cost > 0 else 0
gap_color = "#00ffcc" if gap_pct <= 5 else ("#f2c94c" if gap_pct <= 15 else "#ff3366")
gap_emoji = "🟢" if gap_pct <= 5 else ("🟡" if gap_pct <= 15 else "🔴")

col_d.markdown(f"""
<div class='kpi-card' style='border-left-color:{gap_color};'>
    <div style='color:{gap_color};'>📐 <b>Óptimo Greedy (benchmark)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>${greedy_cost:,.0f}</div>
    <div style='font-size:0.85rem;color:#ccc;'>{gap_emoji} Gap GA: <b>+{gap_pct:.1f}%</b> sobre óptimo</div>
</div>""", unsafe_allow_html=True)

# ── Fórmulas Matemáticas (Ayuda visual para defensa) ─────────────────────────
with st.expander("📐 Ver Formulación Matemática del Modelo (I. de Operaciones)"):
    st.markdown("""
    Este simulador resuelve el siguiente modelo de programación matemática para el Despacho Económico de Carga, 
    evaluado internamente mediante una Función de Penalización Exterior asimétrica:
    """)
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown("**1. Función Objetivo:** Minimizar el costo operativo total")
        st.latex(r"\min f(x) = \sum_{j=1}^{8} \left( x_j \cdot C_j \cdot USD_j \right)")
        st.markdown("**2. Restricción (Demanda):** La potencia debe ser mayor o igual a $D$")
        st.latex(r"g(x): \sum_{j=1}^{8} \left( x_j \cdot C_j \right) \geq D_{Fuzzy}")
    with col_eq2:
        st.markdown("**3. Vector de Decisión:** Fracción de carga del generador $j$")
        st.latex(r"x_j \in [0, 1] \quad \forall j \in \{1,\dots,8\}")
        st.markdown("**4. Función de Fitness con Penalización ($P$):**")
        st.latex(r"Fitness(x) = f(x) + P(x)")
        st.latex(r"P(x) = \begin{cases} 0 & \text{si } g(x) \text{ se cumple} \\ 10^6 + 10^3 \times (\text{déficit}) & \text{en caso contrario} \end{cases}")

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ====================================================================
# 4. ESTADO OPERATIVO DEL CLÚSTER — 8 GEN (2 filas × 4 cols)
# ====================================================================
st.markdown("### 🖥️ Estado Operativo del Clúster de Generación (8 Unidades)")
st.caption(
    "El GA prioriza generadores de menor costo unitario. "
    "Gen 3 (BACKUP·$200/kW) y Gen 6 (EMERGENCIA·$250/kW) solo deben activarse "
    "cuando la demanda supera la capacidad de los eficientes."
)

row1_cols = st.columns(4)
row2_cols = st.columns(4)
all_cols  = list(row1_cols) + list(row2_cols)

ROLES = {
    0:"Carga Media·Alto Costo",  1:"Carga Alta·Mod.",
    2:"BACKUP CRÍTICO·$200/kW",  3:"Base·Eficiente·$80/kW",
    4:"MÁXIMA·Eficiente·$90/kW", 5:"EMERGENCIA·$250/kW",
    6:"Flexible·Mod.",            7:"MÁS EFICIENTE·$70/kW"
}

for i in range(N_GENES):
    carga_pct    = int(best_chrom[i])
    cap_max      = GENERATORS[i, 0]
    costo_kw     = GENERATORS[i, 1]
    kw_aportados = allocation[i]
    costo_gen    = kw_aportados * costo_kw

    is_active     = carga_pct > 0
    s_class       = "gen-active"   if is_active else "gen-inactive"
    i_class       = "active-icon"  if is_active else "inactive-icon"
    icon          = "⚙️"          if is_active else "💤"
    b_class       = "badge-on"    if is_active else "badge-off"
    p_class       = "progress-fill" if is_active else "progress-fill-inactive"

    html = f"""<div class="gen-card {s_class}">
    <div class="gen-icon {i_class}">{icon}</div>
    <div class="gen-title">GEN {i+1}</div>
    <div style="font-size:0.62rem;color:#888;margin-bottom:4px;">{ROLES[i]}</div>
    <div class="gen-badge {b_class}">{"OPERANDO (ON)" if is_active else "APAGADO (OFF)"}</div>
    <div class="gen-data">Potencia: <b>{kw_aportados:.1f} / {int(cap_max)} kW</b></div>
    <div class="cost-data">Costo: ${costo_gen:,.0f}</div>
    <div class="progress-rail">
        <div class="{p_class}" style="width:{carga_pct}%;"></div>
    </div>
    <div style="text-align:right;font-size:0.75rem;color:#00ffcc;font-weight:bold;">{carga_pct}% Carga</div>
    <div class="gen-footer">Eficiencia: ${costo_kw:.0f} USD/kW</div>
</div>"""
    with all_cols[i]:
        st.markdown(html, unsafe_allow_html=True)

# ── Gráfico Visual del Clúster de Generación ──────────────────────────────────
# Este gráfico muestra de forma compacta y comparativa el estado de TODOS los
# generadores simultáneamente: carga asignada (%) como barras horizontales,
# con código de color por tipo de generador (eficiente / moderado / backup).
st.markdown("#### 📊 Gráfico del Clúster — Carga (%) y Potencia (kW) por Generador")

# Construimos los vectores de datos para el gráfico
gen_nombres   = [f"Gen {i+1}  ({int(GENERATORS[i,0])} kW máx)" for i in range(N_GENES)]
cargas_pct    = [int(best_chrom[i]) for i in range(N_GENES)]
kw_vals       = [float(allocation[i]) for i in range(N_GENES)]
costos_unitarios = [float(GENERATORS[i, 1]) for i in range(N_GENES)]

# Paleta de colores según costo unitario del generador:
#   Verde intenso  → muy eficiente (USD/kW bajo)
#   Amarillo       → costo moderado
#   Rojo/naranja   → BACKUP o EMERGENCIA (caro)
def _color_por_costo(costo_unit, activo):
    if not activo:
        return 'rgba(80, 80, 90, 0.5)'   # gris apagado
    if costo_unit <= 90:
        return 'rgba(0, 220, 160, 0.85)'  # verde: eficiente
    elif costo_unit <= 130:
        return 'rgba(100, 180, 255, 0.85)' # azul: moderado
    else:
        return 'rgba(255, 100, 80, 0.85)'  # rojo: caro / backup

colores_barras = [_color_por_costo(costos_unitarios[i], cargas_pct[i] > 0)
                  for i in range(N_GENES)]

# Figura con dos paneles horizontales (subplots lado a lado)
fig_cluster = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Carga Asignada por el AG (%)",
        "Potencia Despachada (kW)",
    ),
    horizontal_spacing=0.12
)

# ── Panel izquierdo: Porcentaje de carga (%) por generador ────────────────
fig_cluster.add_trace(go.Bar(
    y=gen_nombres,                       # eje Y = nombres de generadores
    x=cargas_pct,                        # eje X = % de carga asignada
    orientation='h',                     # barras HORIZONTALES
    name='Carga (%)',
    marker=dict(color=colores_barras, line=dict(color='rgba(255,255,255,0.1)', width=1)),
    text=[f'{v}%' for v in cargas_pct],  # etiqueta al final de cada barra
    textposition='outside',
    textfont=dict(color='#eee', size=11),
    hovertemplate='<b>%{y}</b><br>Carga: %{x}%<extra></extra>',
), row=1, col=1)

# Línea vertical de referencia al 100% (capacidad máxima)
fig_cluster.add_vline(
    x=100, line_dash='dot', line_color='rgba(255,255,255,0.2)',
    row=1, col=1
)

# ── Panel derecho: kW despachados por generador ───────────────────────────
fig_cluster.add_trace(go.Bar(
    y=gen_nombres,
    x=kw_vals,
    orientation='h',
    name='Potencia (kW)',
    marker=dict(color=colores_barras, line=dict(color='rgba(255,255,255,0.1)', width=1)),
    text=[f'{v:.0f} kW' for v in kw_vals],
    textposition='outside',
    textfont=dict(color='#eee', size=11),
    hovertemplate='<b>%{y}</b><br>Potencia: %{x} kW<extra></extra>',
), row=1, col=2)

# Línea vertical indicando la demanda total requerida (referencia proporcional)
# Se dibuja en el panel derecho como guía visual de demanda
fig_cluster.add_vline(
    x=demand_val, line_dash='dash', line_color='#f2c94c',
    annotation_text=f"Demanda: {demand_val:.0f} kW",
    annotation_font_color='#f2c94c',
    annotation_position='top right',
    row=1, col=2
)

# Layout general del gráfico del clúster
fig_cluster.update_layout(
    template='plotly_dark',
    plot_bgcolor='#161b26',
    paper_bgcolor='#0e1117',
    font=dict(color='#ddd'),
    height=360,
    showlegend=False,
    margin=dict(l=10, r=20, t=55, b=10),
    title=dict(
        text=(f"Estado del Clúster Diésel · Demanda FIS: {demand_val:.0f} kW · "
              f"Despacho GA: {final_prod:.0f} kW"),
        x=0.5, xanchor='center', font=dict(size=13, color='#eee')
    ),
)
fig_cluster.update_xaxes(gridcolor='#2a2a3a', range=[0, 115], row=1, col=1)
fig_cluster.update_xaxes(gridcolor='#2a2a3a', row=1, col=2)
fig_cluster.update_yaxes(gridcolor='#2a2a3a')

st.plotly_chart(fig_cluster, use_container_width=True)
st.caption(
    "🟢 Verde: generadores eficientes (≤$90/kW)  |  "
    "🔵 Azul: costo moderado ($91–$130/kW)  |  "
    "🔴 Rojo: BACKUP / EMERGENCIA (>$130/kW)  |  "
    "⬜ Gris: apagado por el optimizador"
)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ====================================================================
# 5. PANEL DE AUDITORÍA MATEMÁTICA
# ====================================================================
st.markdown("### 📊 Panel de Auditoría Matemática")
st.caption("Evidencia de convergencia del GA y conjunto difuso activado — para defensa técnica universitaria.")

c_conv, c_fuzzy = st.columns([1.3, 1])

# ── Gráfica de Convergencia (Costo Limpio solamente) ─────────────────────────
with c_conv:
    actual_gens = len(cost_history)   # puede ser < generations si hubo parada anticipada
    gens_axis   = list(range(1, actual_gens + 1))

    valid_pairs  = [(g, c) for g, c in zip(gens_axis, cost_history) if not np.isnan(c)]
    first_feasible = valid_pairs[0][0] if valid_pairs else None

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=gens_axis, y=cost_history,
        mode='lines', name='Costo Óptimo Válido (USD)',
        line=dict(color='#00ffcc', width=3),
        fill='tozeroy', fillcolor='rgba(0,255,204,0.07)',
        connectgaps=False,   # NaN → gaps naturales = escape del espacio infactible
    ))

    if first_feasible and first_feasible > 2:
        fig_conv.add_vline(
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
    fig_conv.update_layout(
        title=dict(
            text=f"Convergencia GA — Costo Operativo Mínimo · {convergence_note}",
            x=0.5, xanchor='center', font=dict(color='#eee', size=13)
        ),
        xaxis=dict(title="Generación (t)", gridcolor='#2a2a3a', color='#ccc'),
        yaxis=dict(title="Costo USD", gridcolor='#2a2a3a', color='#ccc'),
        template="plotly_dark",
        plot_bgcolor="#161b26", paper_bgcolor="#0e1117",
        font=dict(color='#ddd'),
        legend=dict(orientation='h', y=1.1, font=dict(color='#ddd')),
    )
    st.plotly_chart(fig_conv, use_container_width=True)
    st.caption(
        "ℹ️ Los gaps iniciales (si existen) indican generaciones donde ningún individuo "
        "satisfacía g(x) (espacio infactible). Ilustra el escape de la barrera de penalización."
    )

# ── Gráfica difusa nativa de skfuzzy ─────────────────────────────────────────
with c_fuzzy:
    st.markdown("**Conjunto Difuso Mamdani Defuzzificado (9 reglas AND):**")
    try:
        fig_f = plot_fuzzy_result(demand_sim, demand_var)
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

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ====================================================================
# 6. DESPACHO ECONÓMICO FINAL — COMPARACIÓN GA vs. GREEDY  (SECCIÓN 3)
# ====================================================================
st.markdown("### 📈 Despacho Económico Final — GA vs. Óptimo Greedy")
st.caption(
    "El Greedy es la solución analítica **óptima** para funciones de costo lineales "
    "(Principio de Optimalidad de Bellman). Sirve como referencia para cuantificar la "
    "calidad del GA. Con modelos de costo no-lineales, el Greedy falla y el GA se justifica."
)

gen_labels = [f"Gen {i+1}<br>({int(GENERATORS[i,0])}kW)" for i in range(N_GENES)]

ga_costs_op     = [allocation[i] * GENERATORS[i, 1]      for i in range(N_GENES)]
greedy_costs_op = [greedy_alloc[i] * GENERATORS[i, 1]    for i in range(N_GENES)]

fig_dispatch = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"Potencia Asignada por Gen. (kW)  |  GA={final_prod:.0f} kW  vs  Greedy={greedy_kw:.0f} kW",
        f"Costo por Gen. (USD)  |  GA=${final_cost:,.0f}  vs  Greedy=${greedy_cost:,.0f}",
    ),
    horizontal_spacing=0.1
)

# Colores GA: gradiente verde según % de carga
def _bar_color(pct):
    g = int(255 * (pct / 100))
    return f"rgba(0,{g},min({g}+50,204),0.85)" if pct > 0 else "rgba(60,60,70,0.5)"

bar_colors_ga = [
    f"rgba(0,{int(200*(best_chrom[i]/100))},{int(180*(best_chrom[i]/100))},0.85)"
    if best_chrom[i] > 0 else "rgba(50,50,60,0.5)"
    for i in range(N_GENES)
]

# ── Subplot 1: kW asignados ──────────────────────────────────────────────────
# GA bars
fig_dispatch.add_trace(go.Bar(
    x=gen_labels, y=list(allocation),
    name="GA (kW)", marker=dict(color=bar_colors_ga, line=dict(color='rgba(0,255,204,0.4)', width=1)),
    text=[f"<b>{v:.0f}</b>" for v in allocation],
    textposition='outside', textfont=dict(color='#00ffcc', size=10),
), row=1, col=1)

# Greedy bars
fig_dispatch.add_trace(go.Bar(
    x=gen_labels, y=list(greedy_alloc),
    name="Greedy (kW)", marker=dict(color='rgba(150,180,255,0.4)', line=dict(color='#6699ff', width=1)),
    text=[f"<b>{v:.0f}</b>" for v in greedy_alloc],
    textposition='outside', textfont=dict(color='#6699ff', size=10),
), row=1, col=1)

# Línea de capacidad máxima (limit visible)
fig_dispatch.add_trace(go.Scatter(
    x=gen_labels, y=list(GENERATORS[:, 0]),
    mode='lines+markers', name="Cap. máx.",
    line=dict(color='#ff6b6b', dash='dash', width=2),
    marker=dict(size=6, symbol='diamond'),
), row=1, col=1)

# ── Subplot 2: Costo por generador ───────────────────────────────────────────
fig_dispatch.add_trace(go.Bar(
    x=gen_labels, y=ga_costs_op,
    name="Costo GA (USD)",
    marker=dict(color='#f2c94c', opacity=0.85),
    text=[f"<b>${v:,.0f}</b>" for v in ga_costs_op],
    textposition='outside', textfont=dict(color='#f2c94c', size=10),
), row=1, col=2)

fig_dispatch.add_trace(go.Bar(
    x=gen_labels, y=greedy_costs_op,
    name="Costo Greedy (USD)",
    marker=dict(color='rgba(102,153,255,0.5)', line=dict(color='#6699ff', width=1)),
    text=[f"<b>${v:,.0f}</b>" for v in greedy_costs_op],
    textposition='outside', textfont=dict(color='#6699ff', size=10),
), row=1, col=2)

fig_dispatch.update_layout(
    barmode='group',
    template="plotly_dark",
    plot_bgcolor="#161b26", paper_bgcolor="#0e1117",
    font=dict(color='#ddd'),
    height=440, showlegend=True,
    legend=dict(orientation='h', y=1.14, x=0.25, font=dict(size=11)),
    title=dict(
        text=f"Cromosoma óptimo x* · Costo GA: <b>${final_cost:,.0f}</b> · "
             f"Greedy: <b>${greedy_cost:,.0f}</b> · Gap: <b>{gap_pct:.1f}%</b> {gap_emoji}",
        x=0.5, xanchor='center', font=dict(size=13)
    ),
)
fig_dispatch.update_yaxes(gridcolor='#2a2a3a', row=1, col=1)
fig_dispatch.update_yaxes(gridcolor='#2a2a3a', row=1, col=2)
st.plotly_chart(fig_dispatch, use_container_width=True)

# ── Interpretación académica del GA vs. Greedy ───────────────────────────────
with st.expander("📖 Interpretación académica: ¿Por qué usar GA si existe una solución greedy?"):
    st.markdown(f"""
**Para funciones de costo LINEALES** (como este modelo), el algoritmo greedy es óptimo y resolve en O(N log N).
El GA, con codificación entera y {generations} generaciones, obtiene un costo de **${final_cost:,.0f}** vs.
el óptimo greedy de **${greedy_cost:,.0f}** — un gap de **{gap_pct:.2f}%**.

**¿Por qué implementar el GA entonces?**

1. **Generalidad:** El GA acepta cualquier función f(x) — incluyendo modelos cuadráticos de función de
   calor `Cⱼ(Pⱼ) = aⱼ·Pⱼ² + bⱼ·Pⱼ + cⱼ` (modelo real de generadores diésel) sin cambiar ningún operador.
2. **Restricciones adicionales:** Restricciones de ramping (cambio máximo de carga entre intervalos),
   costos de arranque/parada (Unit Commitment), y restricciones de red eléctrica rompen la separabilidad
   del problema. El GA maneja todo esto solo cambiando la función `evaluate_fitness()`.
3. **Integración con FIS:** El acoplamiento dinámico con el motor difuso permite replanning en tiempo real,
   cosa que la programación lineal no permite sin reformular el modelo matemático completo.
    """)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ====================================================================
# 7. VECTOR DE DECISIÓN ÓPTIMO x* Y COMPARACIÓN CON GREEDY
# ====================================================================
with st.expander("🔬 Cromosoma óptimo x* (vector de decisión GA) vs. Greedy"):
    cols_header = st.columns([2] + [1]*N_GENES)
    cols_header[0].markdown("**Método**")
    for i in range(N_GENES):
        cols_header[i+1].markdown(f"**Gen {i+1}**")

    # Fila GA
    cols_ga = st.columns([2] + [1]*N_GENES)
    cols_ga[0].markdown("🤖 **GA (x\\*)**")
    for i in range(N_GENES):
        color = "#00ffcc" if best_chrom[i] > 0 else "#555"
        cols_ga[i+1].markdown(
            f"<div style='text-align:center;color:{color};font-size:1.3rem;font-weight:bold;'>"
            f"{int(best_chrom[i])}%</div>", unsafe_allow_html=True
        )

    # Fila Greedy
    cols_gr = st.columns([2] + [1]*N_GENES)
    cols_gr[0].markdown("📐 **Greedy (óptimo)**")
    for i in range(N_GENES):
        color = "#6699ff" if greedy_pct[i] > 0 else "#555"
        cols_gr[i+1].markdown(
            f"<div style='text-align:center;color:{color};font-size:1.3rem;font-weight:bold;'>"
            f"{int(greedy_pct[i])}%</div>", unsafe_allow_html=True
        )
