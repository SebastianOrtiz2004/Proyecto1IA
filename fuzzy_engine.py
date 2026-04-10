import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')   # Evita conflictos de backend con Streamlit
import matplotlib.pyplot as plt


def build_fuzzy_system():
    """
    Construye y compila el Sistema de Inferencia Difusa (FIS) de tipo Mamdani.
    Esta función es PURA (sin estado de simulación) y puede cachearse de forma
    segura con @st.cache_resource en app.py para evitar recompilaciones.

    ──────────────────────────────────────────────────────
    JUSTIFICACIÓN MATEMÁTICA (Defensa PhD):
    ──────────────────────────────────────────────────────
    Se modelan la incertidumbre de las variables lingüísticas mediante funciones
    de pertenencia triangulares μ(x) ∈ [0,1].
    El motor opera bajo lógica Mamdani (operadores min/max estrictos):
        T-Norma (AND)    : μ_AND(x,y) = min(μ_A(x), μ_B(y))
        T-Conorma (OR)   : μ_OR(x,y)  = max(μ_A(x), μ_B(y))
    Defuzzificación por Centroide (Centro de Gravedad):
        u* = ∫ u · μ_agregada(u) du  /  ∫ μ_agregada(u) du
    ──────────────────────────────────────────────────────

    ESCENARIO: Planta en modo isla con 8 generadores diésel.
    Capacidad total instalada: ~2950 kW.

    Returns
    -------
    demand_ctrl : ctrl.ControlSystem    — grafo compilado de reglas
    demand_var  : ctrl.Consequent       — variable de salida (para plot)
    """

    # ── 1. UNIVERSOS DE DISCURSO ────────────────────────────────────────────
    # Temperatura de máquinas : 0 a 100 °C
    temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
    # Nivel de producción de la fábrica : 0 a 100 %
    production  = ctrl.Antecedent(np.arange(0, 101, 1), 'production')
    # Demanda energética estimada : 200 a 2200 kW
    # (ajustado al parque de 8 gen. con cap. total ~2950 kW)
    demand = ctrl.Consequent(np.arange(200, 2201, 1), 'demand')

    # ── 2. FUNCIONES DE PERTENENCIA (MF) TRIANGULARES ──────────────────────
    # Antecedente: Temperatura
    #   frio(0,0,45)  normal(30,50,70)  caliente(55,100,100)
    temperature['frio']     = fuzz.trimf(temperature.universe, [0,   0,   45])
    temperature['normal']   = fuzz.trimf(temperature.universe, [30,  50,  70])
    temperature['caliente'] = fuzz.trimf(temperature.universe, [55, 100, 100])

    # Antecedente: Producción
    #   bajo(0,0,45)  medio(30,50,70)  alto(55,100,100)
    production['bajo']  = fuzz.trimf(production.universe, [0,   0,   45])
    production['medio'] = fuzz.trimf(production.universe, [30,  50,  70])
    production['alto']  = fuzz.trimf(production.universe, [55, 100, 100])

    # Consecuente: Demanda  (rango ampliado para 8 generadores)
    #   baja(200,200,800)  media(600,1100,1600)  alta(1300,2200,2200)
    demand['baja']  = fuzz.trimf(demand.universe, [200,  200,   800])
    demand['media'] = fuzz.trimf(demand.universe, [600,  1100, 1600])
    demand['alta']  = fuzz.trimf(demand.universe, [1300, 2200, 2200])

    # ── 3. BASE DE REGLAS MAMDANI (9 reglas — cobertura completa del hiperespacio) ─
    #
    # Tabla de cobertura (3 conjuntos × 3 conjuntos = 9 combinaciones):
    #
    #   Producc. \ Temp. │  Frío         │  Normal       │  Caliente     │
    #   ─────────────────┼───────────────┼───────────────┼───────────────┤
    #   Alto             │ rule1 (alta)  │ rule2 (alta)  │ rule3 (alta)  │
    #   Medio            │ rule4 (baja)  │ rule5 (media) │ rule6 (alta)  │  ← rule6 cubre el gap
    #   Bajo             │ rule7 (baja)  │ rule8 (baja)  │ rule9 (media) │
    #
    # Todas usan T-Norma (AND = min) — coherencia semántica estricta Mamdani.
    # La separación en reglas atómicas (sin OR compuesto) facilita la
    # trazabilidad de la activación y la justificación ante un auditor.
    # ─────────────────────────────────────────────────────────────────────────

    # Bloque 1: Producción ALTA → demanda siempre ALTA (independiente de temp.)
    rule1 = ctrl.Rule(production['alto'] & temperature['frio'],      demand['alta'])
    rule2 = ctrl.Rule(production['alto'] & temperature['normal'],    demand['alta'])
    rule3 = ctrl.Rule(production['alto'] & temperature['caliente'],  demand['alta'])

    # Bloque 2: Producción MEDIA → demanda varía con temperatura
    rule4 = ctrl.Rule(production['medio'] & temperature['frio'],     demand['baja'])
    rule5 = ctrl.Rule(production['medio'] & temperature['normal'],   demand['media'])
    rule6 = ctrl.Rule(production['medio'] & temperature['caliente'], demand['alta'])  # ← regla crítica añadida

    # Bloque 3: Producción BAJA → demanda generalmente BAJA, sube con temperatura
    rule7 = ctrl.Rule(production['bajo'] & temperature['frio'],      demand['baja'])
    rule8 = ctrl.Rule(production['bajo'] & temperature['normal'],    demand['baja'])
    rule9 = ctrl.Rule(production['bajo'] & temperature['caliente'],  demand['media'])

    # ── 4. COMPILACIÓN DEL MOTOR DE INFERENCIA ──────────────────────────────
    # PROPIEDAD DE COMPLETITUD (Mamdani, 1975 / Zadeh, 1973):
    # Una base de reglas es COMPLETA si y solo si para todo punto (x₁,x₂) del
    # universo de entrada, al menos una regla tiene μ_antecedente > 0.
    # Con 9 reglas AND y MFs triangulares solapadas (ver tabla anterior),
    # se garantiza que μ_agregada(u) > 0 para cualquier entrada ∈ [0,100]².
    # → No existe riesgo de ZeroDivisionError en el centroide.
    # → El sistema produce output definido para TODA combinación de entradas.
    demand_ctrl = ctrl.ControlSystem([
        rule1, rule2, rule3,   # producción alta
        rule4, rule5, rule6,   # producción media
        rule7, rule8, rule9,   # producción baja
    ])

    return demand_ctrl, demand


def estimate_demand(demand_ctrl, demand_var, temperature_val: float, production_val: float):
    """
    Fase de Inferencia: recibe el sistema PRE-COMPILADO y valores crisp.
    Aplica np.clip para robustez ante entradas fuera del universo.

    Retorna el valor defuzzificado (centroide) y el objeto simulación
    para posterior visualización del conjunto activado.

    Parameters
    ----------
    demand_ctrl   : ctrl.ControlSystem  — motor compilado (cacheado en app.py)
    demand_var    : ctrl.Consequent     — variable consecuente
    temperature_val : float             — temperatura en °C [0, 100]
    production_val  : float             — nivel de producción en % [0, 100]

    Returns
    -------
    crisp_output : float                — demanda en kW (valor defuzzificado)
    sim          : ControlSystemSimulation — objeto con estado de la inferencia
    """
    sim = ctrl.ControlSystemSimulation(demand_ctrl)

    # Clip defensivo: garantiza que las entradas estén dentro del universo
    sim.input['temperature'] = float(np.clip(temperature_val, 0, 100))
    sim.input['production']  = float(np.clip(production_val,  0, 100))

    # Cómputo del centroide: u* = ∫ u·μ(u) du / ∫ μ(u) du
    sim.compute()

    return sim.output['demand'], sim


def plot_fuzzy_result(demand_sim, demand_var):
    """
    Genera un objeto Figure de Matplotlib con el conjunto difuso del consecuente
    activado según el resultado de la última inferencia.

    skfuzzy dibuja sobre plt.gcf() automáticamente mediante demand_var.view().
    Se recupera la figura instanciada y se devuelve para que app.py la controle
    (evita memory leaks con plt.close() que se hace en app.py).
    """
    demand_var.view(sim=demand_sim)
    fig = plt.gcf()
    fig.set_size_inches(7, 3)
    plt.tight_layout()
    return fig
