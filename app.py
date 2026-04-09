import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Aplicar estética oscura a los gráficos matemáticos de matplotlib
plt.style.use('dark_background')

# Importaciones modulares locales (Arquitectura plana)
from fuzzy_engine import estimate_demand, plot_fuzzy_result
from genetic_optimizer import run_genetic_algorithm, GENERATORS

# ====================================================================
# 1. CONFIGURACIÓN DEL FRAMEWORK & UI EXPERIENCIA PREMIUM
# ====================================================================
st.set_page_config(page_title="Simulador de Planta Diésel", layout="wide", page_icon="⚡", initial_sidebar_state="expanded")

# Inyección de CSS (Glassmorphism & Animaciones Neón Visuales) para representar la planta física
st.markdown("""
<style>
/* Reset base */
.main { background-color: #0e1117; color: white;}

/* Tarjetas métricas superiores */
.kpi-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-left: 5px solid #00ffcc;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.3s;
}
.kpi-card:hover { transform: translateY(-5px); }

/* Tarjetas físicas de simulación de generadores */
.gen-card {
    background: linear-gradient(145deg, rgba(20,25,35,0.9), rgba(15,20,30,0.9));
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 25px 20px;
    margin-bottom: 20px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    position: relative;
    overflow: hidden;
}

/* Efectos visuales de encendido (ON) */
.gen-active {
    border: 1px solid #00ffcc;
    box-shadow: 0 0 25px rgba(0, 255, 204, 0.2), inset 0 0 20px rgba(0, 255, 204, 0.05);
}

/* Efectos visuales de apagado (OFF) */
.gen-inactive {
    border: 1px solid #ff3366;
    opacity: 0.65;
    filter: grayscale(80%);
}

.gen-icon { font-size: 3.5rem; margin-bottom: 15px; }

/* Efecto de Turbina Corriendo */
.active-icon { 
    text-shadow: 0 0 15px #00ffcc, 0 0 30px #00ffcc; 
    animation: generator_pulse 1.5s infinite; 
}
.inactive-icon { color: #444; }

@keyframes generator_pulse {
    0% { transform: scale(1); filter: brightness(1); }
    50% { transform: scale(1.08); filter: brightness(1.3); }
    100% { transform: scale(1); filter: brightness(1); }
}

/* Barras de Tanque/Carga simuladas fisicamente */
.progress-rail {
    background: #1a1d24;
    border-radius: 10px;
    height: 18px;
    width: 100%;
    margin-top: 20px;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00C9FF 0%, #00ffcc 100%);
    border-radius: 10px;
    transition: width 1s cubic-bezier(0.22, 1, 0.36, 1);
    box-shadow: 0 0 10px #00ffcc;
}
.progress-fill-inactive { background: #444; }

/* Tipografía de la simulación */
.gen-title { font-size: 1.4rem; font-weight: 800; letter-spacing: 1px; color: #fff;}
.gen-badge { 
    display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold; margin: 10px 0;
}
.badge-on { background: rgba(0, 255, 204, 0.2); color: #00ffcc; }
.badge-off { background: rgba(255, 51, 102, 0.2); color: #ff3366; }

.gen-data { font-size: 1.1rem; color: #ddd; margin: 5px 0; }
.cost-data { font-size:1.1rem; font-weight: bold; color: #f2c94c; }
.gen-footer { font-size: 0.75rem; color: #888; margin-top: 15px; border-top: 1px dashed #333; padding-top: 10px;}
</style>
""", unsafe_allow_html=True)

# Título Principal
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>⚡ Simulador en Tiempo Real: Planta Diésel</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #aaa; margin-bottom: 30px;'>Optimización Interactiva vía Algoritmo Genético + Lógica Difusa</p>", unsafe_allow_html=True)

# ====================================================================
# 2. TABLERO DE CONTROL (Entradas Interactivas de Simulación)
# ====================================================================
st.sidebar.markdown("### 🎛️ Modificación de Variables (Causa)")
st.sidebar.caption("Al interactuar con estos controles, todo el clúster térmico recalculará la carga distribuida instantáneamente.")

temperature_val = st.sidebar.slider("🌡 Temperatura Externa (°C)", 0.0, 100.0, 60.0, step=1.0)
production_val = st.sidebar.slider("🏭 Carga Productiva de la Fábrica (%)", 0.0, 100.0, 75.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Algoritmo Genético")
st.sidebar.caption("Parámetros de la metaheurística de optimización.")
pop_size = st.sidebar.slider("Tamaño de Población", 10, 200, 60)
generations = st.sidebar.slider("Número de Generaciones", 10, 200, 100)
mutation_rate = st.sidebar.slider("Tasa de Mutación ($\mu$)", 0.01, 0.5, 0.1)

# ====================================================================
# FASE LÓGICA (Ejecución Instantánea de la Red)
# ====================================================================
demand_val, sim, demand_var = estimate_demand(temperature_val, production_val)

# El GA evalúa matemáticamente el óptimo
best_chrom, allocation, final_cost, final_prod, history = run_genetic_algorithm(
    demand=demand_val, pop_size=pop_size, generations=generations, mutation_rate=mutation_rate
)

# ====================================================================
# 3. CONSOLA CENTRAL: DASHBOARD DE RESULTADOS (Efecto Visual)
# ====================================================================
colA, colB, colC = st.columns(3)

# Tarjeta DEMANDA (DIFUSA)
colA.markdown(f"""
<div class='kpi-card'>
    <div style='color: #00ffcc;'>🧠 <b>Requerimiento Energético Estimado (Mamdani)</b></div>
    <div style='font-size: 2.5rem; font-weight: bold; color: #fff;'>{demand_val:.1f} kW</div>
</div>
""", unsafe_allow_html=True)

# Tarjeta SUMINISTRO ALCANZADO (AG)
delta_diff = final_prod - demand_val
if final_prod >= demand_val:
    status_str = f"Satisfecha ✅ (+{delta_diff:.1f} kW Exceso térmico)"
    color_border = "#00ffcc"
else:
    status_str = f"Déficit Grave ❌ ({delta_diff:.1f} kW Penalización activada)"
    color_border = "#ff3366"

colB.markdown(f"""
<div class='kpi-card' style='border-left-color: {color_border};'>
    <div style='color: {color_border};'>⚙️ <b>Corriente Total Suministrada (Genético)</b></div>
    <div style='font-size: 2.5rem; font-weight: bold; color: #fff;'>{final_prod:.1f} kW</div>
    <div style='font-size: 0.95rem; color: #ccc;'>{status_str}</div>
</div>
""", unsafe_allow_html=True)

# Tarjeta COSTO
colC.markdown(f"""
<div class='kpi-card' style='border-left-color: #f2c94c;'>
    <div style='color: #f2c94c;'>💸 <b>Facturación de Combustible Horaria</b></div>
    <div style='font-size: 2.5rem; font-weight: bold; color: #fff;'>${final_cost:,.2f}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #333;' />", unsafe_allow_html=True)

# ====================================================================
# 4. SIMULACIÓN FÍSICA E INTERACTIVA DE LAS TURBINAS DIÉSEL
# ====================================================================
st.markdown("### 🖥️ Estado Operativo del Clúster de Generación")
st.caption("Visualiza el momento exacto en el que el optimizador enciende (ON) o apaga (OFF) un generador. *La barra inferior denota la carga asignada de 0 a 100%.*")

gen_cols = st.columns(4)

graficos_iconos = ["⚙️", "⚙️", "⚙️", "⚙️"] # Iconos industriales sobrios

for i in range(4):
    carga_pct = best_chrom[i]
    cap_max = GENERATORS[i][0]
    costo_kw = GENERATORS[i][1]
    kw_aportados = allocation[i]
    costo_gen = kw_aportados * costo_kw
    
    is_active = float(carga_pct) > 0.0
    
    # Evaluar clases inyectadas de UI CSS
    status_class = "gen-active" if is_active else "gen-inactive"
    icon_class = "active-icon" if is_active else "inactive-icon"
    icon = graficos_iconos[i] if is_active else "💤"
    badge_class = "badge-on" if is_active else "badge-off"
    progress_class = "progress-fill" if is_active else "progress-fill-inactive"
    
    # Composición estelar de HTML para presentar como "Panel de Control Aeronáutico"
    with gen_cols[i]:
        html_str = f"""<div class="gen-card {status_class}">
    <div class="gen-icon {icon_class}">{icon}</div>
    <div class="gen-title">Generador {i+1}</div>
    <div class="gen-badge {badge_class}">{"OPERANDO (ON)" if is_active else "APAGADO (OFF)"}</div>
    <div class="gen-data">Potencia: <b>{kw_aportados:.1f} / {cap_max} kW</b></div>
    <div class="cost-data">Costo Operativo: ${costo_gen:,.0f}</div>
    <div class="progress-rail">
        <div class="{progress_class}" style="width: {carga_pct}%;"></div>
    </div>
    <div style="text-align: right; font-size: 0.8rem; color:#00ffcc; font-weight:bold;">{carga_pct}% Load</div>
    <div class="gen-footer">Eficiencia Térmica: ${costo_kw:.0f} USD/kW</div>
</div>"""
        st.markdown(html_str, unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #333;' />", unsafe_allow_html=True)

# ====================================================================
# 5. AUDITORÍA ACADÉMICA (GRÁFICAS DE INVESTIGACIÓN DE OPERACIONES)
# ====================================================================
st.markdown("### 📊 Panel de Auditoría Matemática y Matrices")
st.caption("Visión teórica para la defensa técnica universitaria de la asignatura.")

c_graf_1, c_graf_2 = st.columns([1.2, 1])

with c_graf_1:
    fig_conv = px.line(x=np.arange(1, generations + 1), y=history, 
                       labels={'x': 'Iteraciones ($t$)', 'y': 'Costo (USD) + Barrera Penal ($)'})
    fig_conv.update_layout(
        title={'text': "Convergencia del Espacio Soluciones (Fitness AG)", 'x':0.5, 'xanchor': 'center'},
        template="plotly_dark", 
        plot_bgcolor="rgba(0,0,0,0.2)", 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig_conv.update_traces(line_color='#00ffcc', line_width=3, fill='tozeroy', fillcolor='rgba(0, 255, 204, 0.1)')
    st.plotly_chart(fig_conv, use_container_width=True)

with c_graf_2:
    st.markdown("**Conjunto Mamdani Defuzzificado:**")
    try:
        # Se genera la figura y se limpia el layout para ajustarse al neón theme
        fig_f = plot_fuzzy_result(sim, demand_var)
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
        st.error(f"Render nativo de skfuzzy inactivo: {e}")
