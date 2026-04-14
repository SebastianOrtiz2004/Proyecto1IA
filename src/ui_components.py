import streamlit as st

# Guia rapida:
# - Este archivo solo dibuja UI (sin logica matematica).
# - Si cambias textos/estilo del dashboard, empieza por aqui.


# ====================================================================
# SECCIÓN 1: ESTILOS GLOBALES (CSS)
# ====================================================================
def inyectar_css():

    st.markdown("""
<style>
.main { background-color: #0e1117; color: white; }

.tarjeta-kpi {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
    border-left: 5px solid #00ffcc; padding: 20px; border-radius: 8px;
    margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    transition: transform 0.3s;
}
.tarjeta-kpi:hover { transform: translateY(-5px); }

.tarjeta-gen {
    background: linear-gradient(145deg,rgba(20,25,35,0.9),rgba(15,20,30,0.9));
    border:1px solid rgba(255,255,255,0.05); border-radius:16px;
    padding:20px 15px; margin-bottom:18px; text-align:center;
    transition:all 0.4s cubic-bezier(0.175,0.885,0.32,1.275);
    position:relative; overflow:hidden;
}
.gen-activo   { border:1px solid #00ffcc; box-shadow:0 0 25px rgba(0,255,204,0.2),inset 0 0 20px rgba(0,255,204,0.05); }
.gen-apagado  { border:1px solid #ff3366; opacity:0.60; filter:grayscale(80%); }

.icono-gen   { font-size:3rem; margin-bottom:12px; }
.icono-activo  { text-shadow:0 0 15px #00ffcc,0 0 30px #00ffcc; animation:pulso_gen 1.5s infinite; }
.icono-apagado { color:#444; }

@keyframes pulso_gen {
    0%  { transform:scale(1);    filter:brightness(1); }
    50% { transform:scale(1.08); filter:brightness(1.3); }
    100%{ transform:scale(1);    filter:brightness(1); }
}

.barra-fondo {
    background:#1a1d24; border-radius:10px; height:14px; width:100%;
    margin-top:15px; box-shadow:inset 0 2px 4px rgba(0,0,0,0.5);
    position:relative; overflow:hidden;
}
.barra-activa {
    height:100%; background:linear-gradient(90deg,#00C9FF 0%,#00ffcc 100%);
    border-radius:10px; transition:width 1s cubic-bezier(0.22,1,0.36,1);
    box-shadow:0 0 10px #00ffcc;
}
.barra-inactiva { background:#444;height:100%;border-radius:10px; }

.titulo-gen   { font-size:1.2rem; font-weight:800; letter-spacing:1px; color:#fff; }
.emblema-gen  { display:inline-block;padding:3px 9px;border-radius:12px;font-size:0.75rem;font-weight:bold;margin:8px 0; }
.emblema-on   { background:rgba(0,255,204,0.2); color:#00ffcc; }
.emblema-off  { background:rgba(255,51,102,0.2); color:#ff3366; }
.dato-gen     { font-size:0.95rem; color:#ddd; margin:4px 0; }
.costo-gen    { font-size:1rem; font-weight:bold; color:#f2c94c; }
.pie-gen      { font-size:0.7rem;color:#888;margin-top:12px;border-top:1px dashed #333;padding-top:8px; }

.caja-referencia {
    background:rgba(242,201,76,0.08); border:1px solid rgba(242,201,76,0.3);
    border-radius:10px; padding:15px 20px; margin-top:10px;
}
</style>
""", unsafe_allow_html=True)


# SECCIÓN 2: ENCABEZADO PRINCIPAL
def renderizar_encabezado():

    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0;'>Simulador de Despacho Económico — Planta Diésel 8 GEN</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center;color:#aaa;margin-bottom:30px;'>"
        "Optimización Interactiva · Algoritmo Genético (Python Puro) + Lógica Difusa Mamdani (Python Puro) "
        "· Capacidad total instalada: <b>2950 kW</b></p>",
        unsafe_allow_html=True
    )


# SECCIÓN 3: TARJETAS KPI — DASHBOARD CENTRAL

def renderizar_tarjetas_kpi(demanda_kw, potencia_ag, costo_ag, costo_voraz):

    col1, col2, col3, col4 = st.columns(4)

    # ── KPI 1: Inyección de Conocimiento (SID) ────────────────────────────────
    col1.markdown(f"""
<div class='tarjeta-kpi'>
    <div style='color:#00ffcc;'><b>Inyección Apriori (Demanda SID)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>{demanda_kw:.1f} kW</div>
    <div style='font-size:0.8rem;color:#888;'>Conocimiento extraído del dataset histórico</div>
</div>""", unsafe_allow_html=True)

    # ── KPI 2: Potencia AG ────────────────────────────────────────────────────
    diferencia  = potencia_ag - demanda_kw
    color_kpi2  = "#00ffcc" if potencia_ag >= demanda_kw else "#ff3366"
    estado      = f"Satisfecha (+{diferencia:.0f} kW)" if diferencia >= 0 else f"Déficit ({diferencia:.0f} kW)"

    col2.markdown(f"""
<div class='tarjeta-kpi' style='border-left-color:{color_kpi2};'>
    <div style='color:{color_kpi2};'><b>Potencia AG (cromosoma óptimo)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>{potencia_ag:.1f} kW</div>
    <div style='font-size:0.85rem;color:#ccc;'>{estado}</div>
</div>""", unsafe_allow_html=True)

    # ── KPI 3: Costo AG ───────────────────────────────────────────────────────
    col3.markdown(f"""
<div class='tarjeta-kpi' style='border-left-color:#f2c94c;'>
    <div style='color:#f2c94c;'><b>Costo AG (USD/h)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>${costo_ag:,.0f}</div>
    <div style='font-size:0.8rem;color:#888;'>Incluye penalización térmica</div>
</div>""", unsafe_allow_html=True)

    # ── KPI 4: Heurística Greedy (Benchmark) ──────────────────────────────────
    brecha_pct  = ((costo_ag - costo_voraz) / costo_voraz * 100) if costo_voraz > 0 else 0
    color_brecha = "#00ffcc" if brecha_pct <= 5 else ("#f2c94c" if brecha_pct <= 15 else "#ff3366")
    emoji_brecha = "BAJA"      if brecha_pct <= 5 else ("MEDIA"      if brecha_pct <= 15 else "ALTA")

    col4.markdown(f"""
<div class='tarjeta-kpi' style='border-left-color:{color_brecha};'>
    <div style='color:{color_brecha};'><b>Heurística Greedy (Benchmark)</b></div>
    <div style='font-size:2.2rem;font-weight:bold;color:#fff;'>${costo_voraz:,.0f}</div>
    <div style='font-size:0.85rem;color:#ccc;'>{emoji_brecha} Diferencial AG: <b>{brecha_pct:+.1f}%</b></div>
</div>""", unsafe_allow_html=True)

    return brecha_pct, emoji_brecha


# SECCIÓN 4: FORMULACIÓN MATEMÁTICA (EXPANDER)
def renderizar_formulas_matematicas():
    """
    Expander colapsable con las fórmulas LaTeX del modelo de optimización:
      - Función objetivo con término térmico cuadrático
      - Restricción de demanda g(x) ≥ D_SID
      - Vector de decisión (x_j ∈ [0,1])
      - Función de aptitud con penalización exterior asimétrica P(x)
    """
    with st.expander("Ver Formulación Matemática del Modelo (I. de Operaciones)"):
        st.markdown("""
    Este simulador resuelve un despacho económico no lineal con costo lineal + térmico.
    En implementación, cada gen se codifica en porcentaje entero (0-100), equivalente a pasos de 1%.
    """)
        col_izq, col_der = st.columns(2)
        with col_izq:
            st.markdown("**1. Función Objetivo:** Minimizar el costo operativo total (con término térmico)")
            st.latex(r"\min f(x,T) = \sum_{j=1}^{8} \left[ base_j \cdot P_j + \alpha_j \cdot \frac{T}{100} \cdot P_j^2 \right]")
            st.markdown("**2. Restricción (Demanda):** La potencia debe ser mayor o igual a $D$")
            st.latex(r"g(x): \sum_{j=1}^{8} P_j \geq D_{SID}")
        with col_der:
            st.markdown("**3. Vector de Decisión (discreto):** Fracción de carga del generador $j$")
            st.latex(r"x_j \in \{0, 0.01, \dots, 1.00\} \quad \forall j \in \{1,\dots,8\}")
            st.latex(r"P_j = x_j \cdot Cap_j")
            st.markdown("**4. Función de Aptitud con Penalización ($P$):**")
            st.latex(r"Aptitud(x) = f(x,T) + P(x)")
            st.latex(r"P(x) = \begin{cases} 0 & \text{si } g(x) \text{ se cumple} \\ 10^6 + 10^3 \times (\text{déficit}) & \text{en caso contrario} \end{cases}")


# SECCIÓN 5: TARJETAS DE ESTADO OPERATIVO (8 GENERADORES)
def renderizar_tarjetas_generadores(mejor_cromosoma, asignacion, GENERADORES, N_GENERADORES):

    # Roles descriptivos de cada generador (índice base-0)
    ROLES = {
        0: "Carga Media · Alto Costo",    1: "Carga Alta · Moderado",
        2: "RESPALDO CRÍTICO · $200/kW",  3: "Base · Eficiente · $80/kW",
        4: "MÁXIMA · Eficiente · $90/kW", 5: "EMERGENCIA · $250/kW",
        6: "Flexible · Moderado",          7: "MÁS EFICIENTE · $70/kW"
    }

    st.markdown(" Estado Operativo del Clúster de Generación (8 Unidades)")
    st.caption(
        "El AG prioriza generadores de menor costo unitario. "
        "Gen 3 (RESPALDO·$200/kW) y Gen 6 (EMERGENCIA·$250/kW) solo deben activarse "
        "cuando la demanda supera la capacidad de los eficientes."
    )

    fila1 = st.columns(4)
    fila2 = st.columns(4)
    todas_columnas = list(fila1) + list(fila2)

    for i in range(N_GENERADORES):
        porcentaje_carga = int(mejor_cromosoma[i])
        capacidad_max    = GENERADORES[i][0]
        costo_por_kw     = GENERADORES[i][1]
        kw_aportados     = asignacion[i]
        costo_parcial    = kw_aportados * costo_por_kw

        esta_activo  = porcentaje_carga > 0
        clase_tarj   = "gen-activo"   if esta_activo else "gen-apagado"
        clase_icono  = "icono-activo" if esta_activo else "icono-apagado"
        icono        = "ACTIVO"       if esta_activo else "INACTIVO"
        clase_emblema = "emblema-on"  if esta_activo else "emblema-off"
        clase_barra   = "barra-activa" if esta_activo else "barra-inactiva"

        html = f"""<div class="tarjeta-gen {clase_tarj}">
    <div class="icono-gen {clase_icono}">{icono}</div>
    <div class="titulo-gen">GEN {i+1}</div>
    <div style="font-size:0.62rem;color:#888;margin-bottom:4px;">{ROLES[i]}</div>
    <div class="emblema-gen {clase_emblema}">{"EN LINEA" if esta_activo else "FUERA DE LINEA"}</div>
    <div class="dato-gen">Potencia: <b>{kw_aportados:.1f} / {int(capacidad_max)} kW</b></div>
    <div class="costo-gen">Costo lineal: ${costo_parcial:,.0f}</div>
    <div class="barra-fondo">
        <div class="{clase_barra}" style="width:{porcentaje_carga}%;"></div>
    </div>
    <div style="text-align:right;font-size:0.75rem;color:#00ffcc;font-weight:bold;">{porcentaje_carga}% Carga</div>
    <div class="pie-gen">Costo unitario base: ${costo_por_kw:.0f} USD/kW</div>
</div>"""
        with todas_columnas[i]:
            st.markdown(html, unsafe_allow_html=True)


# SECCIÓN 6: VECTOR DE DECISIÓN ÓPTIMO x* (EXPANDER)
def renderizar_vector_decision(mejor_cromosoma, porcentaje_voraz, N_GENERADORES):

    with st.expander("Vector de Estado Óptimo (x*) vs. Heurística Greedy"):
        # Fila de encabezados
        cols_encabezado = st.columns([2] + [1]*N_GENERADORES)
        cols_encabezado[0].markdown("**Método**")
        for i in range(N_GENERADORES):
            cols_encabezado[i+1].markdown(f"**Gen {i+1}**")

        # Fila AG
        cols_ag = st.columns([2] + [1]*N_GENERADORES)
        cols_ag[0].markdown("**AG (x\\*)**")
        for i in range(N_GENERADORES):
            color = "#00ffcc" if mejor_cromosoma[i] > 0 else "#555"
            cols_ag[i+1].markdown(
                f"<div style='text-align:center;color:{color};font-size:1.3rem;font-weight:bold;'>"
                f"{int(mejor_cromosoma[i])}%</div>", unsafe_allow_html=True
            )

        # Fila Voraz
        cols_voraz = st.columns([2] + [1]*N_GENERADORES)
        cols_voraz[0].markdown("**Voraz (referencia)**")
        for i in range(N_GENERADORES):
            color = "#6699ff" if porcentaje_voraz[i] > 0 else "#555"
            cols_voraz[i+1].markdown(
                f"<div style='text-align:center;color:{color};font-size:1.3rem;font-weight:bold;'>"
                f"{int(porcentaje_voraz[i])}%</div>", unsafe_allow_html=True
            )


