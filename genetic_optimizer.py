"""
genetic_optimizer.py — Algoritmo Genético de Despacho Económico (PYTHON PURO)
==============================================================================
Implementación DESDE CERO del Algoritmo Genético (AG) y del Despacho Voraz
de referencia, sin ninguna librería científica externa.

Solo se usan módulos de la biblioteca estándar de Python:
  - random : para la inicialización aleatoria, selección por torneo y mutación

NO se utiliza numpy, scipy, ni ninguna librería de cálculo numérico.
Todas las operaciones se realizan con listas nativas y bucles Python.

──────────────────────────────────────────────────────────────────────────────
FUNDAMENTO TEÓRICO (Holland, 1992; Goldberg, 1989):
──────────────────────────────────────────────────────────────────────────────
Basado en el Teorema de los Esquemas (Holland, 1975):
    El AG procesa implícitamente O(N³) bloques de construcción (esquemas)
    con solo N evaluaciones por generación («paralelismo implícito»).
    Esquemas con aptitud superior al promedio reciben crecimiento exponencial
    en generaciones sucesivas → convergencia al óptimo con alta probabilidad.

Cromosoma:  x = [g₁,g₂,g₃,g₄,g₅,g₆,g₇,g₈] ∈ {0,...,100}⁸
Espacio de búsqueda: |S| = 101⁸ ≈ 1.08×10¹⁶  (justifica metaheurística)
──────────────────────────────────────────────────────────────────────────────
"""

import random


# =====================================================================================
# FLOTA DE GENERADORES DIÉSEL — ESCENARIO REAL (Modo Isla, 8 Unidades)
# =====================================================================================
# Formato de cada elemento: [Capacidad Máxima Nominal (kW), Costo Operativo (USD/kW·h)]
# ──────────────────────────────────────────────────────────────────────────────────────
# Estrategia de despacho óptima (para costo lineal):
#   Prioridad de carga ← orden ascendente de costo unitario:
#   Gen 8 (70) → Gen 4 (80) → Gen 5 (90) → Gen 2 (100) →
#   Gen 7 (110) → Gen 1 (150) → Gen 3 (200) → Gen 6 (250)
# ──────────────────────────────────────────────────────────────────────────────────────
# Capacidad total instalada: 300+500+200+400+600+150+450+350 = 2950 kW
GENERADORES = [
    [300.0, 150.0],   # Gen 1: Cap. media,      costo alto
    [500.0, 100.0],   # Gen 2: Cap. alta,        costo moderado
    [200.0, 200.0],   # Gen 3: Cap. baja,        RESPALDO CRÍTICO (costoso)
    [400.0,  80.0],   # Gen 4: Cap. grande,      base de carga eficiente
    [600.0,  90.0],   # Gen 5: Cap. MÁXIMA,      muy eficiente
    [150.0, 250.0],   # Gen 6: Cap. mínima,      EMERGENCIA (el más caro)
    [450.0, 110.0],   # Gen 7: Cap. media-alta,  costo moderado
    [350.0,  70.0],   # Gen 8: Cap. media,       el MÁS EFICIENTE del parque
]

# Número de genes del cromosoma (= número de generadores)
N_GENERADORES: int = len(GENERADORES)   # = 8

# =====================================================================================
# COEFICIENTES DE SENSIBILIDAD TÉRMICA (αⱼ) — Modelo de Costo Cuadrático-Térmico
# =====================================================================================
# La temperatura exterior afecta la eficiencia de cada generador diésel de forma
# diferente según su diseño, sistema de refrigeración y antigüedad.
#
# Modelo de costo extendido:
#   Cⱼ(Pⱼ, T) = base_j × Pⱼ  +  αⱼ × (T/100) × Pⱼ²
#               └── lineal ──┘   └──── cuadrático-térmico ────┘
#
# El término cuadrático NO SEPARA el problema → el Greedy lineal deja de ser óptimo.
# El GA evalúa cromosomas completos → puede descubrir repartos de carga más baratos.
#
# Unidades: αⱼ en [USD / kW²]
# Rango físico razonable: 0.0001 – 0.001 (cuadrático ≤ 10% del costo base)
# ──────────────────────────────────────────────────────────────────────────────────────
COEFICIENTES_TERMICOS = [
    0.00050,   # Gen 1: $150/kW base — sensibilidad media     (refrigeración estándar)
    0.00030,   # Gen 2: $100/kW base — sensibilidad moderada  (carga alta, buen enfriamiento)
    0.00080,   # Gen 3: $200/kW base — sensibilidad ALTA      (respaldo, peor disipación)
    0.00020,   # Gen 4:  $80/kW base — sensibilidad BAJA      (base load, robusto)
    0.00030,   # Gen 5:  $90/kW base — sensibilidad moderada  (alta capacidad)
    0.00100,   # Gen 6: $250/kW base — sensibilidad MÁX.      (emergencia, sin enfriamiento)
    0.00040,   # Gen 7: $110/kW base — sensibilidad moderada  (flexible, rango medio)
    0.00010,   # Gen 8:  $70/kW base — sensibilidad MÍNIMA    (más eficiente, mejor diseño)
]


# =====================================================================================
# FUNCIÓN DE REFERENCIA: DESPACHO VORAZ (BENCHMARK)
# =====================================================================================
def despacho_voraz(demanda: float, temperatura: float = 0.0):
    """
    Algoritmo Voraz (Greedy) de Despacho Económico — Solución de Referencia.

    ──────────────────────────────────────────────────────────────────────────────
    COMPORTAMIENTO CON MODELO LINEAL (temperatura = 0):
    ──────────────────────────────────────────────────────────────────────────────
    Para costo lineal f(x) = Σ base_j · Pⱼ, el Voraz es ÓPTIMO GLOBAL:
        → Ordena por costo base ascendente y asigna carga de mayor a menor.

    ──────────────────────────────────────────────────────────────────────────────
    COMPORTAMIENTO CON MODELO CUADRÁTICO-TÉRMICO (temperatura > 0):
    ──────────────────────────────────────────────────────────────────────────────
    Con costo Cⱼ(Pⱼ,T) = base_j·Pⱼ + αⱼ·(T/100)·Pⱼ², el Voraz ordena por
    costo marginal inicial (dC/dP|_{P=0} = base_j) e intenta asignar al más barato.
    PERO el término cuadrático hace que el costo marginal AUMENTE con la carga:
        dC/dP = base_j + 2·αⱼ·(T/100)·Pⱼ
    El Voraz ignora este aumento → asigna demasiada carga al primer generador y
    rechaza repartos equitativos que serían más baratos. → SOLUCIÓN SUBÓPTIMA.
    El AG no tiene esta limitación: evalúa el costo total del cromosoma completo.
    ──────────────────────────────────────────────────────────────────────────────

    Parámetros
    ----------
    demanda     : float — demanda en kW a satisfacer
    temperatura : float — temperatura exterior en °C [0..100]

    Retorna
    -------
    asignacion   : list[float] (8,) — kW asignados por generador
    costo_total  : float            — costo total (con término térmico si T > 0)
    potencia_kw  : float            — potencia total despachada
    porcentaje   : list[int]   (8,) — porcentaje de carga por generador [0..100]
    """
    # Ordenar por costo base ascendente (el Voraz no puede anticipar el término cuadrático)
    # Índices: [7,3,4,1,6,0,2,5]  →  Gen8(70)→Gen4(80)→Gen5(90)→...→Gen6(250)
    orden_prioridad = sorted(range(N_GENERADORES), key=lambda i: GENERADORES[i][1])

    asignacion = [0.0] * N_GENERADORES
    restante   = demanda

    for indice in orden_prioridad:
        if restante <= 0:
            break
        kw_asignados       = min(restante, GENERADORES[indice][0])
        asignacion[indice] = kw_asignados
        restante          -= kw_asignados

    potencia_kw   = sum(asignacion)
    t_normalizada = temperatura / 100.0

    # Costo con el MISMO modelo térmico-cuadrático que el AG (comparación justa)
    costo_total = sum(
        asignacion[i] * GENERADORES[i][1]
        + COEFICIENTES_TERMICOS[i] * t_normalizada * asignacion[i] ** 2
        for i in range(N_GENERADORES)
    )

    # Porcentaje de carga por generador [0..100] — sin división por cero
    porcentaje = []
    for i in range(N_GENERADORES):
        cap = GENERADORES[i][0]
        if cap > 0:
            p = int(round((asignacion[i] / cap) * 100))
            p = max(0, min(100, p))
        else:
            p = 0
        porcentaje.append(p)

    return asignacion, costo_total, potencia_kw, porcentaje


# =====================================================================================
# EVALUACIÓN DE APTITUD (Función Objetivo + Penalización)
# =====================================================================================
def evaluar_aptitud(poblacion: list, demanda: float, temperatura: float = 0.0):
    """
    Evaluador de la Función Objetivo con Penalización Asimétrica y Costo Térmico.

    ──────────────────────────────────────────────────────────────────
    MODELO DE COSTO CUADRÁTICO-TÉRMICO:
    ──────────────────────────────────────────────────────────────────
    Función de costo extendida:
        Cⱼ(Pⱼ, T) = base_j · Pⱼ  +  αⱼ · (T/100) · Pⱼ²

    Costo total del cromosoma x:
        f(x, T) = Σⱼ [ base_j · Pⱼ  +  αⱼ · (T/100) · Pⱼ² ]

    FORMULACIÓN COMPLETA CON PENALIZACIÓN:
        Aptitud(x) = f(x,T) + P(x)
        P(x) = 0                        si Σ Pⱼ ≥ Demanda
        P(x) = 1×10⁶ + déficit×1000    si Σ Pⱼ < Demanda
    ──────────────────────────────────────────────────────────────────

    Parámetros
    ----------
    poblacion   : list[list[int]] (N, 8) — N cromosomas, cada gen ∈ {0..100}
    demanda     : float                  — demanda en kW (salida del SID Mamdani)
    temperatura : float                  — temperatura exterior en °C [0..100]

    Retorna
    -------
    aptitud      : list[float] (N,)      — aptitud por cromosoma (a minimizar)
    costos       : list[float] (N,)      — costo limpio con término térmico
    potencia_tot : list[float] (N,)      — potencia total por cromosoma
    kw_por_gen   : list[list[float]]     — kW asignados por generador por cromosoma
    """
    t_normalizada = temperatura / 100.0
    aptitud      = []
    costos       = []
    potencia_tot = []
    kw_por_gen   = []

    for cromosoma in poblacion:
        # 1. Convertir alelo [0,100] → kW: Pⱼ = (gⱼ/100) × CapacidadMáxⱼ
        kw = [cromosoma[i] / 100.0 * GENERADORES[i][0] for i in range(N_GENERADORES)]

        # 2. Costo lineal base: Σⱼ(Pⱼ × base_j)
        costo_lineal = sum(kw[i] * GENERADORES[i][1] for i in range(N_GENERADORES))

        # 3. Costo térmico cuadrático: Σⱼ(αⱼ × (T/100) × Pⱼ²)
        costo_termico = sum(
            COEFICIENTES_TERMICOS[i] * t_normalizada * kw[i] ** 2
            for i in range(N_GENERADORES)
        )

        # 4. Costo total con modelo térmico (sin penalización)
        costo = costo_lineal + costo_termico

        # 5. Potencia total del cromosoma
        potencia = sum(kw)

        # 6. Penalización asimétrica exterior:
        #    ① Salto de barrera 1e6 → separa infactibles de factibles
        #    ② Componente déficit×1000 → ordena infactibles por gravedad
        deficit      = demanda - potencia
        penalizacion = (1e6 + deficit * 1000.0) if deficit > 0 else 0.0

        aptitud.append(costo + penalizacion)
        costos.append(costo)
        potencia_tot.append(potencia)
        kw_por_gen.append(kw)

    return aptitud, costos, potencia_tot, kw_por_gen


# =====================================================================================
# ALGORITMO GENÉTICO PRINCIPAL
# =====================================================================================
def ejecutar_ag(
    demanda:        float,
    temperatura:    float = 0.0,
    tam_poblacion:  int   = 60,
    generaciones:   int   = 100,
    tasa_mutacion:  float = 0.10,
    num_elites:     int   = 2
):
    """
    Algoritmo Genético — Python puro (sin librerías externas de cálculo).

    ──────────────────────────────────────────────────────────────────────────────
    OPERADORES IMPLEMENTADOS:
    ──────────────────────────────────────────────────────────────────────────────
    • Selección:    Torneo estricto k=3 (presión selectiva moderada)
    • Cruzamiento:  Un punto aleatorio, Pc = 0.85  [Goldberg, 1989]
    • Mutación:     Uniforme por gen, μ = tasa_mutacion ∈ (0,1)
    • Elitismo:     Los num_elites mejores pasan directamente a la siguiente gen.
                    Garantía: f*(t+1) ≤ f*(t) para todo t  [Whitley, 1989]
    • Parada:       Estancamiento: PACIENCIA generaciones consecutivas sin mejora

    Parámetros
    ----------
    demanda        : float — demanda en kW (del SID Mamdani)
    temperatura    : float — temperatura exterior en °C [0..100]
    tam_poblacion  : int   — número de cromosomas en la población N
    generaciones   : int   — máximo de generaciones t_max
    tasa_mutacion  : float — probabilidad de mutación por gen μ ∈ (0,1)
    num_elites     : int   — cromosomas élite preservados (elitismo)

    Retorna
    -------
    mejor_cromosoma   : list[int]   (8,) — cromosoma óptimo (% de carga por gen.)
    mejor_asignacion  : list[float] (8,) — kW despachados por generador
    mejor_costo       : float            — costo USD con modelo térmico
    mejor_potencia    : float            — potencia total despachada en kW
    historial_aptitud : list[float]      — aptitud mínima por generación
    historial_costo   : list[float]      — costo mínimo limpio (float('nan') si infactible)
    gen_parada        : int              — generación real en que el AG se detuvo
    """
    # ── INICIALIZACIÓN (t=0): Población aleatoria uniforme ───────────────────
    # P₀ ~ U({0,...,100}⁸)^N — cobertura uniforme del hiperespacio de búsqueda
    poblacion = [
        [random.randint(0, 100) for _ in range(N_GENERADORES)]
        for _ in range(tam_poblacion)
    ]

    historial_aptitud: list = []   # aptitud mínima(t) — incluye penalización
    historial_costo:   list = []   # costo mínimo limpio(t) — nan si sin solución válida

    # ── Criterio de parada por estancamiento ────────────────────────────────
    # Si la aptitud mínima no mejora en PACIENCIA generaciones consecutivas,
    # el AG ha convergido: continuar es computacionalmente inútil.
    PACIENCIA            = max(20, generaciones // 5)
    estancamiento        = 0
    mejor_aptitud_global = float('inf')
    gen_parada           = generaciones   # se actualiza si hay parada anticipada

    for gen in range(generaciones):

        # ── FASE A: EVALUACIÓN FENOTÍPICA ────────────────────────────────────
        aptitud, costos, potencia_tot, _ = evaluar_aptitud(poblacion, demanda, temperatura)

        # ── ELITISMO: Preservar los mejores ANTES de modificar la población ──
        # Ordenar ascendente (menor aptitud = individuo más apto)
        indices_ordenados = sorted(range(tam_poblacion), key=lambda i: aptitud[i])
        elites = [poblacion[i][:] for i in indices_ordenados[:num_elites]]

        # ── REGISTRO DE HISTORIALES ─────────────────────────────────────────
        aptitud_actual = aptitud[indices_ordenados[0]]
        historial_aptitud.append(aptitud_actual)

        # Solo registrar costo si existe algún cromosoma factible (potencia ≥ demanda)
        validos = [i for i in range(tam_poblacion) if potencia_tot[i] >= demanda]
        if validos:
            historial_costo.append(min(costos[i] for i in validos))
        else:
            historial_costo.append(float('nan'))   # generación completamente infactible

        # ── DETECCIÓN DE CONVERGENCIA (Criterio de Estancamiento) ───────────
        if aptitud_actual < mejor_aptitud_global - 1e-6:   # mejora significativa
            mejor_aptitud_global = aptitud_actual
            estancamiento = 0
        else:
            estancamiento += 1

        if estancamiento >= PACIENCIA:
            gen_parada = gen + 1       # generación real de convergencia
            break

        # ── FASE B: SELECCIÓN POR TORNEO ESTRICTO (k=3) ─────────────────────
        # ┌───────────────────────────────────────────────────────────────────┐
        # │  Para cada posición i de la nueva población:                     │
        # │    1. Elegir 3 índices al azar (sin reemplazo)                   │
        # │    2. El cromosoma con menor aptitud gana el torneo              │
        # │    3. El ganador ocupa la posición i en la nueva población        │
        # │  Presión selectiva k=3 → intermedia entre k=2 y k=5             │
        # └───────────────────────────────────────────────────────────────────┘
        nueva_poblacion = []
        for _ in range(tam_poblacion):
            combatientes = random.sample(range(tam_poblacion), 3)
            ganador      = min(combatientes, key=lambda i: aptitud[i])
            nueva_poblacion.append(poblacion[ganador][:])

        # ── FASE C: CRUZAMIENTO DE UN PUNTO (Pc = 0.85) ─────────────────────
        # ┌───────────────────────────────────────────────────────────────────┐
        # │  Sea p ∈ {1,...,7} el punto de corte (aleatorio).                │
        # │  Padre₁ = [ G₁ ··· Gₚ | G_{p+1} ··· G₈ ]                       │
        # │  Padre₂ = [ H₁ ··· Hₚ | H_{p+1} ··· H₈ ]                       │
        # │                          ↕ intercambio de colas                  │
        # │  Hijo₁  = [ G₁ ··· Gₚ   H_{p+1} ··· H₈ ]                       │
        # │  Hijo₂  = [ H₁ ··· Hₚ   G_{p+1} ··· G₈ ]                       │
        # └───────────────────────────────────────────────────────────────────┘
        for i in range(num_elites, tam_poblacion - 1, 2):   # élites inmunes
            if random.random() < 0.85:
                punto  = random.randint(1, N_GENERADORES - 1)
                padre1 = nueva_poblacion[i][:]
                padre2 = nueva_poblacion[i + 1][:]
                nueva_poblacion[i]     = padre1[:punto] + padre2[punto:]
                nueva_poblacion[i + 1] = padre2[:punto] + padre1[punto:]

        # ── FASE D: MUTACIÓN UNIFORME POR GEN ───────────────────────────────
        # ┌───────────────────────────────────────────────────────────────────┐
        # │  Si random() < μ  → gen mutado  = randint(0, 100)               │
        # │  Si random() ≥ μ  → gen intacto                                  │
        # │  Élites [:num_elites] → INMUNES a mutación                       │
        # └───────────────────────────────────────────────────────────────────┘
        for i in range(num_elites, tam_poblacion):
            for j in range(N_GENERADORES):
                if random.random() < tasa_mutacion:
                    nueva_poblacion[i][j] = random.randint(0, 100)

        # ── INYECCIÓN DE ÉLITES: Garantiza aptitud no creciente ─────────────
        # Los num_elites mejores de t se copian directamente a t+1.
        # Propiedad: f*(t+1) ≤ f*(t) para todo t  (Whitley, 1989).
        nueva_poblacion[:num_elites] = elites

        poblacion = nueva_poblacion

    # ── POST-CONVERGENCIA: Extraer el cromosoma con menor aptitud ────────────
    aptitud_final, costos_final, potencia_final, asignacion_final = \
        evaluar_aptitud(poblacion, demanda, temperatura)

    indice_mejor = min(range(tam_poblacion), key=lambda i: aptitud_final[i])

    return (
        poblacion[indice_mejor],             # mejor_cromosoma   list[int] (8,)
        asignacion_final[indice_mejor],      # mejor_asignacion  list[float] (8,)
        costos_final[indice_mejor],          # mejor_costo       float (USD)
        potencia_final[indice_mejor],        # mejor_potencia    float (kW)
        historial_aptitud,                   # historial de aptitud mínima
        historial_costo,                     # historial de costo limpio mínimo
        gen_parada,                          # generación real de parada
    )
