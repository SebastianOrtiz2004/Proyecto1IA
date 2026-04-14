"""
app.py — Orquestador Principal del Simulador de Despacho Económico
==================================================================
Versión Master: UI Clásica para Simulación Manual / UI Enriquecida por Fases para Análisis de Dataset.
"""

import streamlit as st
import csv
from pathlib import Path

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
    page_icon="",
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
st.sidebar.markdown("### Modo de Operación")
modo_app = st.sidebar.radio("Seleccionar visualización:", 
                           ["Simulación Manual", "Análisis de Dataset (Apriori)"])

st.sidebar.markdown("---")

if modo_app == "Simulación Manual":
    st.sidebar.markdown("### Variables de Entrada (Antecedentes SID)")
    valor_temperatura = st.sidebar.slider("Temperatura Exterior (°C)", 0.0, 100.0, 30.0, step=1.0)
    valor_produccion  = st.sidebar.slider("Carga Productiva (%)",      0.0, 100.0, 75.0, step=1.0)
else:
    st.sidebar.info("Modo Dataset: El sistema procesará las 100 muestras del histórico usando reglas minadas por Apriori.")
    valor_temperatura = 25.0 
    valor_produccion = 50.0  

st.sidebar.markdown("---")
st.sidebar.markdown("### Parámetros del Algoritmo Genético")
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
    emoji_brecha = "BAJA" if brecha_pct <= 5 else ("MEDIA" if brecha_pct <= 15 else "ALTA")

    # RENDERIZADO UI
    # SECTION 1: INFERENCIA INTELIGENTE (EL CEREBRO)
    st.markdown("### Fase 1: Inferencia Inteligente (Mamdani)")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.write("**Entradas Crisp:**")
        st.metric("Temperatura Ambiente", f"{valor_temperatura}°C")
        st.metric("Carga Industrial", f"{valor_produccion}%")
        st.markdown("---")
        with st.expander("Ver Activacion de Reglas (base completa)"):
            reglas_con_grado = []
            for prod, temp, dem in sistema_control['reglas']:
                grado = min(
                    simulacion_difusa['mu_temp'].get(temp, 0.0),
                    simulacion_difusa['mu_prod'].get(prod, 0.0),
                )
                reglas_con_grado.append((prod, temp, dem, grado))

            reglas_con_grado.sort(key=lambda x: x[3], reverse=True)
            activas = 0
            for prod, temp, dem, grado in reglas_con_grado:
                estado = "ACTIVA" if grado > 0 else "INACTIVA"
                if grado > 0:
                    activas += 1
                st.write(
                    f"SI [T={temp}] Y [C={prod}] ENTONCES [D={dem}] | "
                    f"{estado} | activacion={grado:.2f}"
                )
            st.caption(f"Reglas activas en esta entrada: {activas} de {len(reglas_con_grado)}")
    with c2:
        graficar_membresia_difusa(simulacion_difusa, None, graficar_resultado_difuso)
    
    st.divider()

    # SECTION 2: MECÁNICA EVOLUTIVA (EL MOTOR)
    st.markdown("### Fase 2: Mecánica Evolutiva (Algoritmo Genético)")
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
    st.markdown("### Fase 3: Estado Operativo del Clúster")
    renderizar_tarjetas_kpi(demanda_kw, potencia_ag, costo_ag, costo_voraz)
    st.markdown("---")
    renderizar_tarjetas_generadores(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES)
    
    st.markdown("#### Distribución de Carga y Potencia")
    graficar_cluster_barras(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES, demanda_kw, potencia_ag)
    
    st.divider()

    # SECTION 4: VALIDACIÓN Y BENCHMARK (LA DECISIÓN)
    st.markdown("### Fase 4: Validación y Benchmark Económico")
    
    # Gráficos ocupando el ancho completo
    graficar_cluster_kmeans(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES, potencia_ag, costo_ag, brecha_pct)
    
    st.markdown("#### Comparativa de Despacho: AG vs. Heurística Greedy")
    graficar_comparacion_despacho(asignacion, asignacion_voraz, GENERADORES, N_GENERADORES, potencia_ag, potencia_voraz, costo_ag, costo_voraz, brecha_pct, emoji_brecha, mejor_cromosoma)
    
    renderizar_vector_decision(mejor_cromosoma, porcentaje_voraz, N_GENERADORES)

else:
    # ── VISTA ENRIQUECIDA POR FASES (AUDITORÍA DE DATASET) ────────────────────
    st.markdown("### Pipeline de Auditoría de IA sobre Dataset Histórico")
    
    ruta_csv = Path(__file__).resolve().parent / "DataSet" / "historico_planta.csv"
    resultados = []
    last_sim_fuzzy = None
    last_h_cost = []
    last_g_st = 0
    last_mejor_c = [0] * N_GENERADORES
    last_asig = [0.0] * N_GENERADORES
    last_pot = 0.0
    last_cost = 0.0
    last_d_est = 0.0
    last_costo_voraz = 0.0
    estadisticas_reglas = {
        (prod, temp, dem): {"activaciones": 0, "suma_grado": 0.0}
        for prod, temp, dem in sistema_control['reglas']
    }
    
    with st.spinner("Procesando 100 muestras histórico..."):
        if ruta_csv.exists():
            with open(ruta_csv, mode='r', encoding='utf-8') as f:
                reader = list(csv.DictReader(f))
                for i, fila in enumerate(reader):
                    try:
                        t = float(fila['Temperatura_Ambiente_C'])
                        p = float(fila['Carga_Industrial_Pct'])
                        d_real = float(fila['Demanda_Real_kW'])
                    except (KeyError, TypeError, ValueError):
                        continue
                    d_est, sim_f = estimar_demanda(sistema_control, None, t, p)
                    for prod_r, temp_r, dem_r in sistema_control['reglas']:
                        grado = min(
                            sim_f['mu_temp'].get(temp_r, 0.0),
                            sim_f['mu_prod'].get(prod_r, 0.0),
                        )
                        if grado > 0:
                            clave = (prod_r, temp_r, dem_r)
                            estadisticas_reglas[clave]["activaciones"] += 1
                            estadisticas_reglas[clave]["suma_grado"] += grado

                    mejor_c, asig, cost, pot, h_apt, h_cost, g_st = ejecutar_ag(
                        demanda=d_est, temperatura=t, tam_poblacion=30, 
                        generaciones=30, modo_rafaga=True
                    )
                    resultados.append({'id': i+1, 'temp': t, 'prod': p, 'dem_real': d_real, 'dem_est': d_est, 'costo': cost})
                    last_sim_fuzzy, last_h_cost, last_g_st = sim_f, h_cost, g_st
                    last_mejor_c, last_asig, last_pot, last_cost, last_d_est = mejor_c, asig, pot, cost, d_est
                    _, costo_voraz, _, _ = despacho_voraz(d_est, t)
                    last_costo_voraz = costo_voraz
        else:
            st.error("Archivo historico_planta.csv no encontrado.")

    # Pestañas Auditoría
    tab1, tab2, tab3, tab4 = st.tabs(["Minería", "Inferencia", "Optimización", "Resultados"])

    with tab1:
        st.markdown("#### Conocimiento Minado")
        col_reg, col_dat = st.columns([1, 1.5])
        with col_reg:
            total_muestras = max(1, len(resultados))
            for prod_r, temp_r, dem_r in sistema_control['reglas']:
                stats = estadisticas_reglas[(prod_r, temp_r, dem_r)]
                activaciones = stats["activaciones"]
                porcentaje = (activaciones / total_muestras) * 100
                grado_prom = (stats["suma_grado"] / activaciones) if activaciones > 0 else 0.0
                estado = "ACTIVA" if activaciones > 0 else "INACTIVA"
                st.info(
                    f"SI [T={temp_r}] Y [P={prod_r}] ENTONCES [D={dem_r}] | "
                    f"{estado} | activaciones={activaciones}/{total_muestras} ({porcentaje:.1f}%) | "
                    f"grado_prom={grado_prom:.2f}"
                )
        with col_dat: st.dataframe(resultados[:10], use_container_width=True)

    with tab2:
        st.markdown("#### Comportamiento del Motor Difuso")
        if last_sim_fuzzy is not None:
            graficar_membresia_difusa(last_sim_fuzzy, None, graficar_resultado_difuso)
        else:
            st.warning("No hay simulaciones válidas para visualizar en este momento.")

    with tab3:
        st.markdown("#### Eficiencia Evolutiva")
        if last_h_cost:
            graficar_convergencia(last_h_cost, last_g_st, 30)
        else:
            st.warning("No hay historial evolutivo disponible.")

    with tab4:
        st.markdown("#### Dashboard Consolidad")
        if resultados:
            renderizar_tarjetas_kpi(last_d_est, last_pot, last_cost, last_costo_voraz)
            # Métricas de validación para defensa: error del SID y desempeño AG.
            errores_abs = [abs(r['dem_real'] - r['dem_est']) for r in resultados]
            mae_sid = sum(errores_abs) / len(errores_abs) if errores_abs else 0.0
            mape_sid = (
                sum((abs(r['dem_real'] - r['dem_est']) / r['dem_real']) for r in resultados if r['dem_real'] != 0)
                / max(1, sum(1 for r in resultados if r['dem_real'] != 0))
            ) * 100
            brechas = []
            for r in resultados:
                _, costo_voraz_muestra, _, _ = despacho_voraz(r['dem_est'], r['temp'])
                if costo_voraz_muestra > 0:
                    brechas.append(((r['costo'] - costo_voraz_muestra) / costo_voraz_muestra) * 100)
            brecha_prom = (sum(brechas) / len(brechas)) if brechas else 0.0
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("MAE SID", f"{mae_sid:.1f} kW")
            c_b.metric("MAPE SID", f"{mape_sid:.1f}%")
            c_c.metric("Brecha promedio AG vs Voraz", f"{brecha_prom:+.1f}%")
            renderizar_tarjetas_generadores(last_mejor_c, last_asig, GENERADORES, N_GENERADORES)
            graficar_analisis_dataset(resultados)
        else:
            st.warning("No se pudieron procesar filas válidas del dataset.")
