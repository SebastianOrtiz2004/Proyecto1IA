"""
ui_components.py — Módulo de Componentes de Interfaz de Usuario
================================================================
Contiene las funciones que renderizan los elementos visuales HTML/CSS
de la aplicación. Cada función es responsable de una sección específica
de la UI, lo que permite ubicar y modificar componentes de forma independiente.

Funciones:
    inject_css()            → Inyecta el CSS global (glassmorphism, animaciones)
    render_header()         → Título y subtítulo principal de la app
    render_kpi_cards()      → 4 tarjetas KPI del dashboard central
    render_math_formulas()  → Expander con fórmulas LaTeX del modelo matemático
    render_generator_cards()→ 8 tarjetas de estado operativo de cada generador
    render_decision_vector()→ Tabla comparativa GA vs. Greedy (cromosoma x*)
    render_academic_expander() → Expander de justificación académica del GA
"""

import streamlit as st
import numpy as np


# ====================================================================
# SECCIÓN 1: ESTILOS GLOBALES (CSS)
# ====================================================================
def inject_css():
    """
    Inyecta el CSS global de la aplicación: glassmorphism, animaciones
    neón, tarjetas de generadores, barras de progreso y badges de estado.
    Se llama UNA sola vez al inicio de app.py antes de renderizar nada.
    """
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


# ====================================================================
# SECCIÓN 2: ENCABEZADO PRINCIPAL
# ====================================================================
def render_header():
    """
    Renderiza el título principal (h1) y el subtítulo descriptivo de la app.
    Incluye la capacidad total instalada del parque de generadores.
    """
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
# SECCIÓN 3: TARJETAS KPI — DASHBOARD CENTRAL
# ====================================================================
def render_kpi_cards(demand_val, final_prod, final_cost, greedy_cost):
    """
    Renderiza las 4 tarjetas del dashboard principal:
      - KPI 1: Demanda estimada por el FIS Mamdani (kW)
      - KPI 2: Potencia total despachada por el GA (kW) y estado (OK/déficit)
      - KPI 3: Costo operativo del GA (USD/h)
      - KPI 4: Costo óptimo Greedy + Gap porcentual vs GA

    Parámetros
    ----------
    demand_val   : float — demanda en kW calculada por el FIS
    final_prod   : float — potencia total del cromosoma óptimo del GA (kW)
    final_cost   : float — costo USD del cromosoma óptimo (sin penalización)
    greedy_cost  : float — costo USD de la solución Greedy (óptimo analítico)

    Retorna
    -------
    gap_pct   : float — gap porcentual GA vs Greedy (para usar en otros módulos)
    gap_emoji : str   — emoji de color según calidad del gap (🟢/🟡/🔴)
    """
    col_a, col_b, col_c, col_d = st.columns(4)

    # ── KPI 1: Demanda FIS ────────────────────────────────────────────
    col_a.markdown(f"""
<div class='kpi-card'>
    <div style='color:#00ffcc;'>🧠 <b>Demanda Estimada (FIS Mamdani)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>{demand_val:.1f} kW</div>
    <div style='font-size:0.8rem;color:#888;'>Centroide defuzzificado · 9 reglas AND</div>
</div>""", unsafe_allow_html=True)

    # ── KPI 2: Potencia GA ────────────────────────────────────────────
    delta   = final_prod - demand_val
    color_b = "#00ffcc" if final_prod >= demand_val else "#ff3366"
    status  = f"Satisfecha ✅ (+{delta:.0f} kW)" if delta >= 0 else f"Déficit ❌ ({delta:.0f} kW)"

    col_b.markdown(f"""
<div class='kpi-card' style='border-left-color:{color_b};'>
    <div style='color:{color_b};'>⚙️ <b>Potencia GA (cromosoma óptimo)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>{final_prod:.1f} kW</div>
    <div style='font-size:0.85rem;color:#ccc;'>{status}</div>
</div>""", unsafe_allow_html=True)

    # ── KPI 3: Costo GA ───────────────────────────────────────────────
    col_c.markdown(f"""
<div class='kpi-card' style='border-left-color:#f2c94c;'>
    <div style='color:#f2c94c;'>💸 <b>Costo GA (USD/h)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>${final_cost:,.0f}</div>
    <div style='font-size:0.8rem;color:#888;'>Sin penalización</div>
</div>""", unsafe_allow_html=True)

    # ── KPI 4: Benchmark Greedy ───────────────────────────────────────
    gap_pct   = ((final_cost - greedy_cost) / greedy_cost * 100) if greedy_cost > 0 else 0
    gap_color = "#00ffcc" if gap_pct <= 5 else ("#f2c94c" if gap_pct <= 15 else "#ff3366")
    gap_emoji = "🟢"      if gap_pct <= 5 else ("🟡"      if gap_pct <= 15 else "🔴")

    col_d.markdown(f"""
<div class='kpi-card' style='border-left-color:{gap_color};'>
    <div style='color:{gap_color};'>📐 <b>Óptimo Greedy (benchmark)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>${greedy_cost:,.0f}</div>
    <div style='font-size:0.85rem;color:#ccc;'>{gap_emoji} Gap GA: <b>+{gap_pct:.1f}%</b> sobre óptimo</div>
</div>""", unsafe_allow_html=True)

    return gap_pct, gap_emoji


# ====================================================================
# SECCIÓN 4: FORMULACIÓN MATEMÁTICA (EXPANDER)
# ====================================================================
def render_math_formulas():
    """
    Expander colapsable con las fórmulas LaTeX del modelo de optimización:
      - Función objetivo (minimizar costo operativo)
      - Restricción de demanda (g(x) ≥ D_Fuzzy)
      - Vector de decisión (x_j ∈ [0,1])
      - Función de fitness con penalización exterior asimétrica P(x)
    """
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


# ====================================================================
# SECCIÓN 5: TARJETAS DE ESTADO OPERATIVO (8 GENERADORES)
# ====================================================================
def render_generator_cards(best_chrom, allocation, GENERATORS, N_GENES):
    """
    Renderiza las 8 tarjetas de estado operativo del clúster de generadores
    en una cuadrícula de 2 filas × 4 columnas.

    Cada tarjeta muestra:
      - Ícono animado (⚙️ activo / 💤 apagado)
      - Nombre del generador y su rol en el parque
      - Badge ON/OFF con color semántico
      - Potencia asignada / máxima (kW)
      - Costo parcial (USD)
      - Barra de progreso de carga (%)
      - Eficiencia unitaria (USD/kW)

    Parámetros
    ----------
    best_chrom  : ndarray (8,) — genes del cromosoma óptimo (% de carga)
    allocation  : ndarray (8,) — kW asignados por generador
    GENERATORS  : ndarray (8,2) — [capacidad kW, costo USD/kW] por generador
    N_GENES     : int           — número de generadores (= 8)
    """
    # Roles descriptivos de cada generador (índice base-0)
    ROLES = {
        0: "Carga Media·Alto Costo",   1: "Carga Alta·Mod.",
        2: "BACKUP CRÍTICO·$200/kW",   3: "Base·Eficiente·$80/kW",
        4: "MÁXIMA·Eficiente·$90/kW",  5: "EMERGENCIA·$250/kW",
        6: "Flexible·Mod.",             7: "MÁS EFICIENTE·$70/kW"
    }

    st.markdown("### 🖥️ Estado Operativo del Clúster de Generación (8 Unidades)")
    st.caption(
        "El GA prioriza generadores de menor costo unitario. "
        "Gen 3 (BACKUP·$200/kW) y Gen 6 (EMERGENCIA·$250/kW) solo deben activarse "
        "cuando la demanda supera la capacidad de los eficientes."
    )

    row1_cols = st.columns(4)
    row2_cols = st.columns(4)
    all_cols  = list(row1_cols) + list(row2_cols)

    for i in range(N_GENES):
        carga_pct    = int(best_chrom[i])
        cap_max      = GENERATORS[i, 0]
        costo_kw     = GENERATORS[i, 1]
        kw_aportados = allocation[i]
        costo_gen    = kw_aportados * costo_kw

        is_active = carga_pct > 0
        s_class   = "gen-active"   if is_active else "gen-inactive"
        i_class   = "active-icon"  if is_active else "inactive-icon"
        icon      = "⚙️"          if is_active else "💤"
        b_class   = "badge-on"    if is_active else "badge-off"
        p_class   = "progress-fill" if is_active else "progress-fill-inactive"

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


# ====================================================================
# SECCIÓN 6: VECTOR DE DECISIÓN ÓPTIMO x* (EXPANDER)
# ====================================================================
def render_decision_vector(best_chrom, greedy_pct, N_GENES):
    """
    Expander que muestra la tabla comparativa del cromosoma óptimo del GA
    (vector de decisión x*) vs. la solución Greedy, generador por generador.

    El color del valor indica si el generador está activo (verde/azul)
    o apagado (gris). Permite verificar visualmente si el GA está priorizando
    los generadores correctos (los de menor costo unitario).

    Parámetros
    ----------
    best_chrom  : ndarray (8,) — genes óptimos del GA (% de carga por gen.)
    greedy_pct  : ndarray (8,) — porcentajes del despacho Greedy
    N_GENES     : int           — número de generadores
    """
    with st.expander("🔬 Cromosoma óptimo x* (vector de decisión GA) vs. Greedy"):
        # Fila de encabezados
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


# ====================================================================
# SECCIÓN 7: JUSTIFICACIÓN ACADÉMICA (EXPANDER)
# ====================================================================
def render_academic_expander(final_cost, greedy_cost, gap_pct, generations):
    """
    Expander con la justificación académica de por qué se usa un GA
    cuando existe una solución Greedy óptima para costos lineales.

    Explica:
      1. Generalidad del GA para costos no-lineales (función cuadrática)
      2. Manejo de restricciones adicionales (ramping, Unit Commitment)
      3. Integración dinámica con el FIS en tiempo real

    Parámetros
    ----------
    final_cost  : float — costo USD del GA
    greedy_cost : float — costo USD del Greedy
    gap_pct     : float — gap porcentual GA vs Greedy
    generations : int   — número de generaciones configuradas
    """
    with st.expander("📖 Interpretación académica: ¿Por qué usar GA si existe una solución greedy?"):
        st.markdown(f"""
**Para funciones de costo LINEALES** (como este modelo), el algoritmo greedy es óptimo y resuelve en O(N log N).
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
