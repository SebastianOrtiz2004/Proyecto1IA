"""
app.py — Orquestador Principal del Simulador de Despacho Económico
==================================================================
Punto de entrada de la aplicación Streamlit. Su único rol es:
  1. Configurar la página y el caché del FIS Mamdani
  2. Leer los controles del sidebar (entradas del usuario)
  3. Ejecutar la lógica: FIS → GA → Greedy Benchmark
  4. Llamar a los módulos de UI y graficas para renderizar los resultados

NO contiene lógica de cálculo ni código HTML/CSS. Toda la presentación
está en ui_components.py y charts.py; toda la lógica en fuzzy_engine.py
y genetic_optimizer.py.

Arquitectura del proyecto:
  app.py              → Orquestador (este archivo)
  fuzzy_engine.py     → Motor de Inferencia Difusa Mamdani (FIS)
  genetic_optimizer.py→ Algoritmo Genético vectorizado en NumPy
  ui_components.py    → Componentes HTML/CSS de la interfaz
  charts.py           → Gráficos Plotly y Matplotlib
"""

import streamlit as st
import matplotlib
matplotlib.use('Agg')   # Backend sin ventana para evitar errores de threading
import matplotlib.pyplot as plt

# ── Módulos propios del proyecto ──────────────────────────────────────────────
from fuzzy_engine import build_fuzzy_system, estimate_demand, plot_fuzzy_result
from genetic_optimizer import (
    run_genetic_algorithm, greedy_dispatch, GENERATORS, N_GENES
)
from ui_components import (
    inject_css,
    render_header,
    render_kpi_cards,
    render_math_formulas,
    render_generator_cards,
    render_decision_vector,
    render_academic_expander,
)
from charts import (
    plot_cluster_bars,
    plot_cluster_kmeans,
    plot_convergence,
    plot_fuzzy_membership,
    plot_dispatch_comparison,
)

# ====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y CACHÉ DEL FIS
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
    El FIS Mamdani se compila UNA sola vez y se almacena en caché global.
    @st.cache_resource garantiza que el grafo de reglas no se reconstruya
    en cada interacción del usuario. Reduce latencia ~500ms → ~5ms.
    """
    return build_fuzzy_system()


# ── Inyectar CSS global y renderizar encabezado ───────────────────────────────
inject_css()
render_header()

# ====================================================================
# 2. SIDEBAR — Controles de Simulación (Entradas del Usuario)
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
# 3. FASE LÓGICA — FIS → GA → GREEDY BENCHMARK
# ====================================================================

# ── Paso 1: Sistema de Inferencia Difusa (FIS Mamdani) ───────────────────────
# Fuzzifica Temperatura y Producción, aplica las 9 reglas AND de la base
# de conocimiento, y devuelve la demanda estimada en kW (centroide).
demand_ctrl, demand_var = get_fuzzy_system()
demand_val, demand_sim  = estimate_demand(demand_ctrl, demand_var, temperature_val, production_val)

# ── Paso 2: Algoritmo Genético (meta-heurística de optimización) ───────────────────────
# Recibe la demanda del FIS como restricción dura Y la temperatura exterior como
# parámetro del modelo de costo cuadrático-térmico:
#   Cⱼ(Pⱼ, T) = base_j × Pⱼ  +  αⱼ × (T/100) × Pⱼ²
# El término cuadrático rompe la separabilidad del problema → el Greedy YA NO ES
# óptimo. El GA evalúa el cromosoma completo y puede encontrar repartos de carga
# que el Greedy descarta por operar generador a generador.
best_chrom, allocation, final_cost, final_prod, fitness_history, cost_history, gen_stopped = \
    run_genetic_algorithm(
        demand=demand_val,
        temperature=temperature_val,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=mutation_rate,
        elite_k=elite_k,
    )

# ── Paso 3: Benchmark Greedy (heurística de solución de referencia) ────────────────
# Ordena generadores por costo BASE ascendente y asigna carga al más barato
# primero. Con T=0 (sin penalización térmica), es el óptimo global.
# Con T>0, el Greedy NO VE el término cuadrático en su decisión de asignación
# → da solución SUBÓPTIMA. Esto justifica el GA con modelo de costo no-lineal.
greedy_alloc, greedy_cost, greedy_kw, greedy_pct = greedy_dispatch(demand_val, temperature_val)

# ====================================================================
# 4. RENDERIZADO DE LA INTERFAZ (llamadas a los módulos de UI)
# ====================================================================

# ── Sección A: Dashboard KPI (4 tarjetas métricas) ───────────────────────────
gap_pct, gap_emoji = render_kpi_cards(demand_val, final_prod, final_cost, greedy_cost)

# ── Sección B: Formulación matemática del modelo (expander colapsable) ────────
render_math_formulas()

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección C: Estado operativo de los 8 generadores (tarjetas HTML) ─────────
render_generator_cards(best_chrom, allocation, GENERATORS, N_GENES)

# ── Sección D: Gráfico de barras del clúster (% carga y kW por generador) ────
plot_cluster_bars(best_chrom, allocation, GENERATORS, N_GENES, demand_val, final_prod)

# ── Sección E: Gráfico de clúster estilo K-Means (nube de puntos) ────────────
plot_cluster_kmeans(best_chrom, allocation, GENERATORS, N_GENES, final_prod, final_cost, gap_pct)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección F: Panel de Auditoría Matemática (convergencia + difuso) ─────────
st.markdown("### 📊 Panel de Auditoría Matemática")
st.caption("Evidencia de convergencia del GA y conjunto difuso activado — para defensa técnica universitaria.")

c_conv, c_fuzzy = st.columns([1.3, 1])

with c_conv:
    plot_convergence(cost_history, gen_stopped, generations)

with c_fuzzy:
    plot_fuzzy_membership(demand_sim, demand_var, plot_fuzzy_result)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección G: Comparación GA vs. Greedy (barras dobles por generador) ────────
plot_dispatch_comparison(
    allocation, greedy_alloc, GENERATORS, N_GENES,
    final_prod, greedy_kw, final_cost, greedy_cost,
    gap_pct, gap_emoji, best_chrom
)

# ── Sección H: Justificación académica del GA (expander colapsable) ───────────
render_academic_expander(final_cost, greedy_cost, gap_pct, generations)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección I: Vector de decisión x* — GA vs. Greedy (tabla comparativa) ──────
render_decision_vector(best_chrom, greedy_pct, N_GENES)
