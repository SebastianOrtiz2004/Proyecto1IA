import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')   # Evita conflictos de hilo con Streamlit
import matplotlib.pyplot as plt


def construir_sistema_difuso():
    """
    Construye y compila el Sistema de Inferencia Difusa (SID) de tipo Mamdani.
    Esta función es PURA (sin estado de simulación) y puede almacenarse en
    caché con @st.cache_resource en app.py para evitar recompilaciones.

    ──────────────────────────────────────────────────────
    JUSTIFICACIÓN MATEMÁTICA (Defensa Universitaria):
    ──────────────────────────────────────────────────────
    Se modela la incertidumbre de las variables lingüísticas mediante funciones
    de pertenencia triangulares μ(x) ∈ [0,1].
    El motor opera bajo lógica Mamdani (operadores mín/máx estrictos):
        T-Norma (AND)    : μ_AND(x,y) = mín(μ_A(x), μ_B(y))
        T-Conorma (OR)   : μ_OR(x,y)  = máx(μ_A(x), μ_B(y))
    Defuzzificación por Centroide (Centro de Gravedad):
        u* = ∫ u · μ_agregada(u) du  /  ∫ μ_agregada(u) du
    ──────────────────────────────────────────────────────

    ESCENARIO: Planta en modo isla con 8 generadores diésel.
    Capacidad total instalada: ~2950 kW.

    Retorna
    -------
    sistema_control : ctrl.ControlSystem  — grafo compilado de reglas
    var_demanda     : ctrl.Consequent     — variable de salida (para graficar)
    """

    # ── 1. UNIVERSOS DE DISCURSO ─────────────────────────────────────────────
    # Temperatura exterior de la planta: 0 a 100 °C
    var_temperatura = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
    # Nivel de carga productiva de la fábrica: 0 a 100 %
    var_produccion  = ctrl.Antecedent(np.arange(0, 101, 1), 'production')
    # Demanda energética estimada: 200 a 2200 kW
    # (ajustado al parque de 8 generadores con capacidad total ~2950 kW)
    var_demanda = ctrl.Consequent(np.arange(200, 2201, 1), 'demand')

    # ── 2. FUNCIONES DE PERTENENCIA TRIANGULARES ─────────────────────────────
    # Antecedente 1: Temperatura
    #   frio(0,0,45)  normal(30,50,70)  caliente(55,100,100)
    var_temperatura['frio']     = fuzz.trimf(var_temperatura.universe, [0,   0,   45])
    var_temperatura['normal']   = fuzz.trimf(var_temperatura.universe, [30,  50,  70])
    var_temperatura['caliente'] = fuzz.trimf(var_temperatura.universe, [55, 100, 100])

    # Antecedente 2: Producción
    #   bajo(0,0,45)  medio(30,50,70)  alto(55,100,100)
    var_produccion['bajo']  = fuzz.trimf(var_produccion.universe, [0,   0,   45])
    var_produccion['medio'] = fuzz.trimf(var_produccion.universe, [30,  50,  70])
    var_produccion['alto']  = fuzz.trimf(var_produccion.universe, [55, 100, 100])

    # Consecuente: Demanda (rango ampliado para 8 generadores)
    #   baja(200,200,800)  media(600,1100,1600)  alta(1300,2200,2200)
    var_demanda['baja']  = fuzz.trimf(var_demanda.universe, [200,  200,   800])
    var_demanda['media'] = fuzz.trimf(var_demanda.universe, [600,  1100, 1600])
    var_demanda['alta']  = fuzz.trimf(var_demanda.universe, [1300, 2200, 2200])

    # ── 3. BASE DE REGLAS MAMDANI (9 reglas — cobertura completa) ────────────
    #
    # Tabla de cobertura (3 conjuntos × 3 conjuntos = 9 combinaciones):
    #
    #   Producción \ Temp. │  Frío           │  Normal         │  Caliente       │
    #   ───────────────────┼─────────────────┼─────────────────┼─────────────────┤
    #   Alto               │ regla1 (alta)   │ regla2 (alta)   │ regla3 (alta)   │
    #   Medio              │ regla4 (baja)   │ regla5 (media)  │ regla6 (alta)   │
    #   Bajo               │ regla7 (baja)   │ regla8 (baja)   │ regla9 (media)  │
    #
    # Todas usan T-Norma (AND = mín) — coherencia semántica estricta Mamdani.
    # ──────────────────────────────────────────────────────────────────────────

    # Bloque 1: Producción ALTA → demanda siempre ALTA (independiente de temp.)
    regla1 = ctrl.Rule(var_produccion['alto'] & var_temperatura['frio'],      var_demanda['alta'])
    regla2 = ctrl.Rule(var_produccion['alto'] & var_temperatura['normal'],    var_demanda['alta'])
    regla3 = ctrl.Rule(var_produccion['alto'] & var_temperatura['caliente'],  var_demanda['alta'])

    # Bloque 2: Producción MEDIA → demanda varía con temperatura
    regla4 = ctrl.Rule(var_produccion['medio'] & var_temperatura['frio'],     var_demanda['baja'])
    regla5 = ctrl.Rule(var_produccion['medio'] & var_temperatura['normal'],   var_demanda['media'])
    regla6 = ctrl.Rule(var_produccion['medio'] & var_temperatura['caliente'], var_demanda['alta'])

    # Bloque 3: Producción BAJA → demanda generalmente baja, sube con temperatura
    regla7 = ctrl.Rule(var_produccion['bajo'] & var_temperatura['frio'],      var_demanda['baja'])
    regla8 = ctrl.Rule(var_produccion['bajo'] & var_temperatura['normal'],    var_demanda['baja'])
    regla9 = ctrl.Rule(var_produccion['bajo'] & var_temperatura['caliente'],  var_demanda['media'])

    # ── 4. COMPILACIÓN DEL MOTOR DE INFERENCIA ───────────────────────────────
    # PROPIEDAD DE COMPLETITUD (Mamdani, 1975 / Zadeh, 1973):
    # Con 9 reglas AND y funciones de pertenencia triangulares solapadas,
    # se garantiza μ_agregada(u) > 0 para cualquier entrada ∈ [0,100]².
    # → No existe riesgo de división por cero en el centroide.
    sistema_control = ctrl.ControlSystem([
        regla1, regla2, regla3,   # producción alta
        regla4, regla5, regla6,   # producción media
        regla7, regla8, regla9,   # producción baja
    ])

    return sistema_control, var_demanda


def estimar_demanda(sistema_control, var_demanda, valor_temperatura: float, valor_produccion: float):
    """
    Fase de Inferencia Difusa: recibe el motor PRE-COMPILADO y valores crisp.
    Aplica np.clip para robustez ante entradas fuera del universo de discurso.

    Retorna el valor defuzzificado (centroide) y el objeto de simulación
    para posterior visualización del conjunto activado.

    Parámetros
    ----------
    sistema_control  : ctrl.ControlSystem        — motor compilado (cacheado)
    var_demanda      : ctrl.Consequent            — variable consecuente
    valor_temperatura : float                     — temperatura en °C [0, 100]
    valor_produccion  : float                     — carga productiva en % [0, 100]

    Retorna
    -------
    salida_crisp : float                          — demanda en kW (centroide)
    simulacion   : ControlSystemSimulation        — objeto con estado de la inferencia
    """
    simulacion = ctrl.ControlSystemSimulation(sistema_control)

    # Recorte defensivo: garantiza que las entradas estén dentro del universo
    simulacion.input['temperature'] = float(np.clip(valor_temperatura, 0, 100))
    simulacion.input['production']  = float(np.clip(valor_produccion,  0, 100))

    # Cómputo del centroide: u* = ∫ u·μ(u) du / ∫ μ(u) du
    simulacion.compute()

    return simulacion.output['demand'], simulacion


def graficar_resultado_difuso(simulacion, var_demanda):
    """
    Genera un objeto Figure de Matplotlib con el conjunto difuso del consecuente
    activado según el resultado de la última inferencia.

    skfuzzy dibuja sobre plt.gcf() automáticamente mediante var_demanda.view().
    Se recupera la figura instanciada y se devuelve para que app.py la controle
    (evitar pérdidas de memoria con plt.close() que se llama en app.py).
    """
    var_demanda.view(sim=simulacion)
    figura = plt.gcf()
    figura.set_size_inches(7, 3)
    plt.tight_layout()
    return figura
