"""
app.py — Orquestador Principal del Simulador de Despacho Económico
==================================================================
Punto de entrada de la aplicación Streamlit. Su único rol es:
  1. Configurar la página y el caché del Sistema de Inferencia Difusa (SID)
  2. Leer los controles del panel lateral (entradas del usuario)
  3. Ejecutar la lógica: SID → AG → Despacho Voraz (referencia)
  4. Llamar a los módulos de interfaz y gráficos para renderizar resultados

NO contiene lógica de cálculo ni código HTML/CSS. Toda la presentación
está en componentes_ui.py y graficos.py; toda la lógica en motor_difuso.py
y optimizador_genetico.py.

Arquitectura del proyecto:
  app.py                → Orquestador (este archivo)
  fuzzy_engine.py       → Motor de Inferencia Difusa Mamdani (SID)
  genetic_optimizer.py  → Algoritmo Genético vectorizado en NumPy
  ui_components.py      → Componentes HTML/CSS de la interfaz
  charts.py             → Gráficos Plotly y Matplotlib
"""

import streamlit as st
import matplotlib
matplotlib.use('Agg')   # Modo sin ventana para evitar errores de hilo
import matplotlib.pyplot as plt

# ── Módulos propios del proyecto ──────────────────────────────────────────────
from fuzzy_engine import (
    construir_sistema_difuso,
    estimar_demanda,
    graficar_resultado_difuso,
)
from genetic_optimizer import (
    ejecutar_ag,
    despacho_voraz,
    GENERADORES,
    N_GENERADORES,
)
from ui_components import (
    inyectar_css,
    renderizar_encabezado,
    renderizar_tarjetas_kpi,
    renderizar_formulas_matematicas,
    renderizar_tarjetas_generadores,
    renderizar_vector_decision,
    renderizar_expander_academico,
)
from charts import (
    graficar_cluster_barras,
    graficar_cluster_kmeans,
    graficar_convergencia,
    graficar_membresia_difusa,
    graficar_comparacion_despacho,
)

# ====================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y CACHÉ DEL SID
# ====================================================================
st.set_page_config(
    page_title="Simulador Planta Diésel — 8 Generadores",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def obtener_sistema_difuso():
    """
    El SID Mamdani se compila UNA sola vez y se almacena en caché global.
    @st.cache_resource garantiza que el grafo de reglas no se reconstruya
    en cada interacción del usuario. Reduce latencia ~500ms → ~5ms.
    """
    return construir_sistema_difuso()


# ── Inyectar CSS global y renderizar encabezado ───────────────────────────────
inyectar_css()
renderizar_encabezado()

# ====================================================================
# 2. PANEL LATERAL — Controles de Simulación (Entradas del Usuario)
# ====================================================================
st.sidebar.markdown("### 🎛️ Variables de Entrada (Antecedentes SID)")
st.sidebar.caption("Cada cambio re-ejecuta el SID y el AG instantáneamente.")

valor_temperatura = st.sidebar.slider("🌡 Temperatura Exterior (°C)", 0.0, 100.0, 60.0, step=1.0)
valor_produccion  = st.sidebar.slider("🏭 Carga Productiva (%)",      0.0, 100.0, 75.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Parámetros del Algoritmo Genético")
st.sidebar.caption("Metaheurística de optimización combinatoria.")

tam_poblacion  = st.sidebar.slider("Tamaño de Población (N)",    10, 200, 60)
num_generaciones = st.sidebar.slider("Generaciones máx. (t_max)", 10, 300, 120)
tasa_mutacion  = st.sidebar.slider("Tasa de Mutación (μ)",      0.01, 0.50, 0.10, step=0.01)
num_elites     = st.sidebar.slider("Élites preservados (k)",      1,   5,   2)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Espacio de búsqueda:** `|S| = 101⁸ ≈ 1.08×10¹⁶`  \n"
    f"**Paciencia (PACIENCIA):** `{max(20, num_generaciones//5)}` gen. sin mejora"
)
st.sidebar.caption(
    "La magnitud del espacio justifica el uso de metaheurísticas en lugar de "
    "fuerza bruta o enumeración exhaustiva."
)

# ====================================================================
# 3. FASE LÓGICA — SID → AG → VORAZ (REFERENCIA)
# ====================================================================

# ── Paso 1: Sistema de Inferencia Difusa Mamdani ─────────────────────────────
# Fuzzifica Temperatura y Producción, aplica las 9 reglas AND de la base
# de conocimiento, y retorna la demanda estimada en kW (centroide).
sistema_control, var_demanda = obtener_sistema_difuso()
demanda_kw, simulacion_difusa = estimar_demanda(
    sistema_control, var_demanda, valor_temperatura, valor_produccion
)

# ── Paso 2: Algoritmo Genético (metaheurística de optimización) ──────────────
# Recibe la demanda del SID como restricción dura Y la temperatura exterior
# como parámetro del modelo de costo cuadrático-térmico:
#   Cⱼ(Pⱼ, T) = base_j × Pⱼ  +  αⱼ × (T/100) × Pⱼ²
# El término cuadrático rompe la separabilidad → el Voraz deja de ser óptimo.
# El AG evalúa el cromosoma completo y descubre repartos de carga óptimos.
mejor_cromosoma, asignacion, costo_ag, potencia_ag, \
    historial_aptitud, historial_costo, gen_parada = ejecutar_ag(
        demanda=demanda_kw,
        temperatura=valor_temperatura,
        tam_poblacion=tam_poblacion,
        generaciones=num_generaciones,
        tasa_mutacion=tasa_mutacion,
        num_elites=num_elites,
    )

# ── Paso 3: Despacho Voraz (heurística de referencia) ────────────────────────
# Ordena generadores por costo BASE ascendente y asigna carga al más barato.
# Con T=0 es el óptimo global; con T>0 da solución subóptima porque no ve
# el término cuadrático → mayor Gap con el AG. Eso justifica el AG.
asignacion_voraz, costo_voraz, potencia_voraz, porcentaje_voraz = \
    despacho_voraz(demanda_kw, valor_temperatura)

# ====================================================================
# 4. RENDERIZADO DE LA INTERFAZ (llamadas a los módulos de UI)
# ====================================================================

# ── Sección A: Dashboard KPI (4 tarjetas métricas) ───────────────────────────
brecha_pct, emoji_brecha = renderizar_tarjetas_kpi(
    demanda_kw, potencia_ag, costo_ag, costo_voraz
)

# ── Sección B: Formulación matemática del modelo (expander colapsable) ────────
renderizar_formulas_matematicas()

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección C: Estado operativo de los 8 generadores (tarjetas HTML) ─────────
renderizar_tarjetas_generadores(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES)

# ── Sección D: Gráfico de barras del clúster (% carga y kW por generador) ────
graficar_cluster_barras(
    mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES, demanda_kw, potencia_ag
)

# ── Sección E: Gráfico de clúster estilo K-Medias (nube de puntos) ────────────
graficar_cluster_kmeans(
    mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES,
    potencia_ag, costo_ag, brecha_pct
)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección F: Panel de Auditoría Matemática (convergencia + difuso) ──────────
st.markdown("### 📊 Panel de Auditoría Matemática")
st.caption(
    "Evidencia de convergencia del AG y conjunto difuso activado — "
    "para defensa técnica universitaria."
)

col_convergencia, col_difuso = st.columns([1.3, 1])

with col_convergencia:
    graficar_convergencia(historial_costo, gen_parada, num_generaciones)

with col_difuso:
    graficar_membresia_difusa(simulacion_difusa, var_demanda, graficar_resultado_difuso)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección G: Comparación AG vs. Voraz (barras dobles por generador) ─────────
graficar_comparacion_despacho(
    asignacion, asignacion_voraz, GENERADORES, N_GENERADORES,
    potencia_ag, potencia_voraz, costo_ag, costo_voraz,
    brecha_pct, emoji_brecha, mejor_cromosoma
)

# ── Sección H: Justificación académica del AG (expander colapsable) ───────────
renderizar_expander_academico(costo_ag, costo_voraz, brecha_pct, num_generaciones)

st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)

# ── Sección I: Vector de decisión x* — AG vs. Voraz (tabla comparativa) ───────
renderizar_vector_decision(mejor_cromosoma, porcentaje_voraz, N_GENERADORES)
