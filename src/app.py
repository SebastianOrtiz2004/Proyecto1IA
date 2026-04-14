"""
app.py — Orquestador Principal del Simulador de Despacho Económico
==================================================================
Versión Master: UI Clásica para Simulación Manual / UI Enriquecida por Fases para Análisis de Dataset.
"""

import streamlit as st
import csv
import os

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
    valor_temperatura = st.sidebar.slider("🌡 Temperatura Exterior (°C)", 0.0, 100.0, 30.0, step=1.0)
    valor_produccion  = st.sidebar.slider("🏭 Carga Productiva (%)",      0.0, 100.0, 75.0, step=1.0)
else:
    st.sidebar.info("Modo Dataset: El sistema procesará las 100 muestras del histórico usando reglas minadas por Apriori.")
    valor_temperatura = 25.0 
    valor_produccion = 50.0  

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Parámetros del Algoritmo Genético")
tam_poblacion  = st.sidebar.slider("Tamaño de Población (N)",    10, 200, 60)
num_generaciones = st.sidebar.slider("Generaciones máx. (t_max)", 10, 300, 120)
tasa_mutacion  = st.sidebar.slider("Tasa de Mutación (μ)",      0.01, 0.50, 0.10, step=0.01)
num_elites     = st.sidebar.slider("Élites preservados (k)",      1,   5,   2)

# ── LOGICA COMPARTIDA ────────────────────────────────────────────────────────
sistema_control, _ = obtener_sistema_difuso()

# ====================================================================
# 3. RENDERIZADO SEGÚN MODO SELECCIONADO
# ====================================================================

if modo_app == "Simulación Manual":
    # ── VISTA CLÁSICA UNIFICADA ──────────────────────────────────────────────
    
    # 1. Ejecución SID
    demanda_kw, simulacion_difusa = estimar_demanda(sistema_control, None, valor_temperatura, valor_produccion)
    
    # 2. Ejecución AG
    mejor_cromosoma, asignacion, costo_ag, potencia_ag, \
        historial_aptitud, historial_costo, gen_parada = ejecutar_ag(
            demanda=demanda_kw, temperatura=valor_temperatura,
            tam_poblacion=tam_poblacion, generaciones=num_generaciones,
            tasa_mutacion=tasa_mutacion, num_elites=num_elites,
        )
    
    # 3. Heurística Greedy (Benchmark)
    asignacion_voraz, costo_voraz, potencia_voraz, porcentaje_voraz = despacho_voraz(demanda_kw, valor_temperatura)

    # 4. Cálculo de Brecha (Necesario para gráficos de Fase 4)
    brecha_pct = ((costo_ag - costo_voraz) / costo_voraz * 100) if costo_voraz > 0 else 0
    emoji_brecha = "🟢" if brecha_pct <= 5 else ("🟡" if brecha_pct <= 15 else "🔴")

    # RENDERIZADO UI
    # SECTION 1: INFERENCIA INTELIGENTE (EL CEREBRO)
    st.markdown("### 🧠 Fase 1: Inferencia Inteligente (Mamdani)")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.write("**Entradas Crisp:**")
        st.metric("Temperatura Ambiente", f"{valor_temperatura}°C")
        st.metric("Carga Industrial", f"{valor_produccion}%")
        st.markdown("---")
        with st.expander("🔍 Ver Reglas de Asociación Activas"):
            for r in sistema_control['reglas_detalles']:
                st.info(f"**SI** [T={r['temp']}] **Y** [C={r['prod']}] **ENTONCES** [D={r['dem']}]")
    with c2:
        graficar_membresia_difusa(simulacion_difusa, None, graficar_resultado_difuso)
    
    st.divider()

    # SECTION 2: MECÁNICA EVOLUTIVA (EL MOTOR)
    st.markdown("### 🧬 Fase 2: Mecánica Evolutiva (Algoritmo Genético)")
    renderizar_formulas_matematicas()
    c3, c4 = st.columns([1, 1.4])
    with c3:
        st.markdown("**Métricas de Convergencia:**")
        st.metric("Generaciones", gen_parada)
        st.metric("Población Total", tam_poblacion)
        st.caption(f"Espacio de Búsqueda: 101^8")
    with c4:
        graficar_convergencia(historial_costo, gen_parada, num_generaciones)

    st.divider()

    # SECTION 3: ESTADO OPERATIVO (EL RESULTADO)
    st.markdown("### ⚡ Fase 3: Estado Operativo del Clúster")
    renderizar_tarjetas_kpi(demanda_kw, potencia_ag, costo_ag, costo_voraz)
    st.markdown("---")
    renderizar_tarjetas_generadores(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES)
    
    st.markdown("#### 📊 Distribución de Carga y Potencia")
    graficar_cluster_barras(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES, demanda_kw, potencia_ag)
    
    st.divider()

    # SECTION 4: VALIDACIÓN Y BENCHMARK (LA DECISIÓN)
    st.markdown("### 📈 Fase 4: Validación y Benchmark Económico")
    
    # Gráficos ocupando el ancho completo
    graficar_cluster_kmeans(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES, potencia_ag, costo_ag, brecha_pct)
    
    st.markdown("#### Comparativa de Despacho: AG vs. Heurística Greedy")
    graficar_comparacion_despacho(asignacion, asignacion_voraz, GENERADORES, N_GENERADORES, potencia_ag, potencia_voraz, costo_ag, costo_voraz, brecha_pct, emoji_brecha, mejor_cromosoma)
    
    renderizar_vector_decision(mejor_cromosoma, porcentaje_voraz, N_GENERADORES)

else:
    # ── VISTA ENRIQUECIDA POR FASES (AUDITORÍA DE DATASET) ────────────────────
    st.markdown("### 📑 Pipeline de Auditoría de IA sobre Dataset Histórico")
    
    ruta_csv = "src/DataSet/historico_planta.csv"
    resultados = []
    
    with st.spinner("Procesando 100 muestras histórico..."):
        if os.path.exists(ruta_csv):
            with open(ruta_csv, mode='r', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
                for i, fila in enumerate(reader):
                    t, p, d_real = float(fila['Temperatura_Ambiente_C']), float(fila['Carga_Industrial_Pct']), float(fila['Demanda_Real_kW'])
                    d_est, sim_f = estimar_demanda(sistema_control, None, t, p)
                    mejor_c, asig, cost, pot, h_apt, h_cost, g_st = ejecutar_ag(
                        demanda=d_est, temperatura=t, tam_poblacion=30, 
                        generaciones=30, modo_rafaga=True
                    )
                    resultados.append({'id': i+1, 'temp': t, 'prod': p, 'dem_real': d_real, 'dem_est': d_est, 'costo': cost})
                    if i == len(reader) - 1:
                        last_sim_fuzzy, last_h_cost, last_g_st = sim_f, h_cost, g_st
                        last_mejor_c, last_asig, last_pot, last_cost, last_d_est = mejor_c, asig, pot, cost, d_est
        else:
            st.error("Archivo historico_planta.csv no encontrado.")

    # Pestañas Auditoría
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Minería", "🧠 Inferencia", "🧬 Optimización", "⚡ Resultados"])

    with tab1:
        st.markdown("#### Conocimiento Minado")
        col_reg, col_dat = st.columns([1, 1.5])
        with col_reg:
            for r in sistema_control['reglas_detalles']: st.info(f"SI [T={r['temp']}] Y [P={r['prod']}] ENTONCES [D={r['dem']}]")
        with col_dat: st.dataframe(resultados[:10], use_container_width=True)

    with tab2:
        st.markdown("#### Comportamiento del Motor Difuso")
        graficar_membresia_difusa(last_sim_fuzzy, None, graficar_resultado_difuso)

    with tab3:
        st.markdown("#### Eficiencia Evolutiva")
        graficar_convergencia(last_h_cost, last_g_st, 30)

    with tab4:
        st.markdown("#### Dashboard Consolidad")
        brecha_v, emoji_v = renderizar_tarjetas_kpi(last_d_est, last_pot, last_cost, 0) # Simplificado para dataset
        renderizar_tarjetas_generadores(last_mejor_c, last_asig, GENERADORES, N_GENERADORES)
        graficar_analisis_dataset(resultados)
