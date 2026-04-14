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

# ── Módulos propios del proyecto ──────────────────────────────────────────────
from LogicaDifusa import (
    construir_sistema_difuso,
    estimar_demanda,
    graficar_resultado_difuso,
)
from AlgoritmoGenetico import (
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
)
from charts import (
    graficar_cluster_barras,
    graficar_cluster_kmeans,
    graficar_convergencia,
    graficar_membresia_difusa,
    graficar_comparacion_despacho,
    graficar_analisis_dataset,
)
import csv
import os

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
# 2. PANEL LATERAL — Controles y Modo de Operación
# ====================================================================
st.sidebar.markdown("### 📊 Modo de Operación")
modo_app = st.sidebar.radio("Seleccionar visualización:", 
                           ["Simulación Manual", "Análisis de Dataset (Apriori)"])

st.sidebar.markdown("---")

if modo_app == "Simulación Manual":
    st.sidebar.markdown("### 🎛️ Variables de Entrada (Antecedentes SID)")
    st.sidebar.caption("Cada cambio re-ejecuta el SID y el AG instantáneamente.")
    valor_temperatura = st.sidebar.slider("🌡 Temperatura Exterior (°C)", 0.0, 100.0, 60.0, step=1.0)
    valor_produccion  = st.sidebar.slider("🏭 Carga Productiva (%)",      0.0, 100.0, 75.0, step=1.0)
else:
    st.sidebar.info("Modo Dataset: El sistema procesará las 100 muestras del histórico usando reglas minadas por Apriori.")
    valor_temperatura = 25.0 # default
    valor_produccion = 50.0  # default

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Parámetros del Algoritmo Genético")
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
# 3. PROCESAMIENTO SEGÚN MODO
# ====================================================================

sistema_control, _ = obtener_sistema_difuso()

if modo_app == "Simulación Manual":
    # ── MODO MANUAL: SID → AG → VORAZ ─────────────────────────────────────────
    demanda_kw, simulacion_difusa = estimar_demanda(
        sistema_control, None, valor_temperatura, valor_produccion
    )

    mejor_cromosoma, asignacion, costo_ag, potencia_ag, \
        historial_aptitud, historial_costo, gen_parada = ejecutar_ag(
            demanda=demanda_kw, temperatura=valor_temperatura,
            tam_poblacion=tam_poblacion, generaciones=num_generaciones,
            tasa_mutacion=tasa_mutacion, num_elites=num_elites,
        )

    asignacion_voraz, costo_voraz, potencia_voraz, porcentaje_voraz = \
        despacho_voraz(demanda_kw, valor_temperatura)

    # RENDERIZADO MANUAL
    brecha_pct, emoji_brecha = renderizar_tarjetas_kpi(demanda_kw, potencia_ag, costo_ag, costo_voraz)
    
    with st.expander("🔍 Ver Reglas Minadas por Apriori"):
        st.write("Estas son las reglas que el motor está usando actualmente (extraídas del dataset):")
        for r in sistema_control['reglas_detalles']:
            st.markdown(f"- **SI** [Temp={r['temp']}] **Y** [Prod={r['prod']}] **ENTONCES** [Demanda={r['dem']}] (Confianza: {r['confianza']:.2f})")

    renderizar_formulas_matematicas()
    st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)
    renderizar_tarjetas_generadores(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES)
    graficar_cluster_barras(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES, demanda_kw, potencia_ag)
    graficar_cluster_kmeans(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES, potencia_ag, costo_ag, brecha_pct)
    st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Panel de Auditoría Matemática")
    col_convergencia, col_difuso = st.columns([1.3, 1])
    with col_convergencia: graficar_convergencia(historial_costo, gen_parada, num_generaciones)
    with col_difuso: graficar_membresia_difusa(simulacion_difusa, None, graficar_resultado_difuso)
    
    st.markdown("<hr style='border:1px solid #333;'/>", unsafe_allow_html=True)
    graficar_comparacion_despacho(asignacion, asignacion_voraz, GENERADORES, N_GENERADORES, potencia_ag, potencia_voraz, costo_ag, costo_voraz, brecha_pct, emoji_brecha, mejor_cromosoma)
    renderizar_vector_decision(mejor_cromosoma, porcentaje_voraz, N_GENERADORES)

else:
    # ── MODO DATASET: PROCESAR 100 MUESTRAS ────────────────────────────────────
    st.markdown("### 📈 Análisis de Desempeño sobre Dataset Histórico")
    st.caption("Procesando 100 registros con reglas minadas por Apriori...")
    
    ruta_csv = "DataSet/historico_planta.csv"
    if not os.path.exists(ruta_csv):
        st.error("Archivo DataSet/historico_planta.csv no encontrado.")
    else:
        resultados = []
        progreso = st.progress(0)
        
        with open(ruta_csv, mode='r') as f:
            reader = list(csv.DictReader(f))
            total = len(reader)
            
            for i, fila in enumerate(reader):
                t = float(fila['Temperatura_Ambiente_C'])
                p = float(fila['Carga_Industrial_Pct'])
                d_real = float(fila['Demanda_Real_kW'])
                
                # SID -> demanda estimada
                d_est, _ = estimar_demanda(sistema_control, None, t, p)
                
                # Ejecutar AG (versión rápida para dataset)
                mejor_c, _, costo, pot, _, _, _ = ejecutar_ag(
                    demanda=d_est, temperatura=t,
                    tam_poblacion=tam_poblacion, generaciones=num_generaciones,
                    modo_rafaga=True
                )
                
                resultados.append({
                    'id': i+1, 'temp': t, 'prod': p, 'dem_real': d_real,
                    'dem_est': d_est, 'costo': costo, 'pot_ag': pot
                })
                progreso.progress((i + 1) / total)
        
        # Dashboard de Dataset
        costo_total = sum(r['costo'] for r in resultados)
        error_medio = sum(abs(r['dem_est'] - r['dem_real']) for r in resultados) / len(resultados)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Costo Acumulado (Simulado)", f"${costo_total:,.0f}")
        c2.metric("Error Medio SID vs Real", f"{error_medio:.1f} kW")
        c3.metric("Reglas Minadas (Apriori)", len(sistema_control['reglas_detalles']))
        
        # Gráficas de Dataset
        graficar_analisis_dataset(resultados)
        
        with st.expander("📄 Ver Tabla de Resultados del Dataset"):
            st.table(resultados[:10]) # Mostrar solo los 10 primeros por legibilidad
            st.caption("Mostrando los primeros 10 registros de 100.")
