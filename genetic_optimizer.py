import numpy as np

# =====================================================================================
# FLOTA DE GENERADORES DIÉSEL — ESCENARIO REAL (Modo Isla, 8 Unidades)
# =====================================================================================
# Columna 0: Capacidad Máxima Nominal  [kW]
# Columna 1: Costo Operativo Directo   [USD / kW·h]
# ──────────────────────────────────────────────────────────────────────────────────────
# Estrategia de despacho óptima (para costo lineal):
#   Prioridad de carga ← orden ascendente de costo unitario:
#   Gen 8 (70) → Gen 4 (80) → Gen 5 (90) → Gen 2 (100) →
#   Gen 7 (110) → Gen 1 (150) → Gen 3 (200) → Gen 6 (250)
# ──────────────────────────────────────────────────────────────────────────────────────
# Capacidad total instalada: 300+500+200+400+600+150+450+350 = 2950 kW
GENERADORES = np.array([
    [300.0, 150.0],   # Gen 1: Cap. media,      costo alto
    [500.0, 100.0],   # Gen 2: Cap. alta,        costo moderado
    [200.0, 200.0],   # Gen 3: Cap. baja,        RESPALDO CRÍTICO (costoso)
    [400.0,  80.0],   # Gen 4: Cap. grande,      base de carga eficiente
    [600.0,  90.0],   # Gen 5: Cap. MÁXIMA,      muy eficiente
    [150.0, 250.0],   # Gen 6: Cap. mínima,      EMERGENCIA (el más caro)
    [450.0, 110.0],   # Gen 7: Cap. media-alta,  costo moderado
    [350.0,  70.0],   # Gen 8: Cap. media,       el MÁS EFICIENTE del parque
])

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
COEFICIENTES_TERMICOS = np.array([
    0.00050,   # Gen 1: $150/kW base — sensibilidad media     (refrigeración estándar)
    0.00030,   # Gen 2: $100/kW base — sensibilidad moderada  (carga alta, buen enfriamiento)
    0.00080,   # Gen 3: $200/kW base — sensibilidad ALTA      (respaldo, peor disipación)
    0.00020,   # Gen 4:  $80/kW base — sensibilidad BAJA      (base load, robusto)
    0.00030,   # Gen 5:  $90/kW base — sensibilidad moderada  (alta capacidad)
    0.00100,   # Gen 6: $250/kW base — sensibilidad MÁX.      (emergencia, sin enfriamiento)
    0.00040,   # Gen 7: $110/kW base — sensibilidad moderada  (flexible, rango medio)
    0.00010,   # Gen 8:  $70/kW base — sensibilidad MÍNIMA    (más eficiente, mejor diseño)
])


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
    asignacion   : ndarray (8,) — kW asignados por generador
    costo_total  : float        — costo total (con término térmico si T > 0)
    potencia_kw  : float        — potencia total despachada
    porcentaje   : ndarray (8,) — porcentaje de carga por generador [0..100]
    """
    # El Voraz ordena por costo base (no puede anticipar el término cuadrático)
    orden_prioridad = np.argsort(GENERADORES[:, 1])   # índices: [7,3,4,1,6,0,2,5]

    asignacion = np.zeros(N_GENERADORES)
    restante   = demanda

    for indice in orden_prioridad:
        if restante <= 0:
            break
        kw_asignados        = min(restante, GENERADORES[indice, 0])
        asignacion[indice]  = kw_asignados
        restante           -= kw_asignados

    potencia_kw = float(np.sum(asignacion))

    # Costo con el MISMO modelo térmico-cuadrático que el AG (comparación justa)
    t_normalizada = temperatura / 100.0
    costo_total   = float(
        np.sum(asignacion * GENERADORES[:, 1])                        # término lineal
        + np.sum(COEFICIENTES_TERMICOS * t_normalizada * asignacion**2)  # término cuadrático
    )

    with np.errstate(divide='ignore', invalid='ignore'):
        porcentaje = np.where(
            GENERADORES[:, 0] > 0,
            np.round((asignacion / GENERADORES[:, 0]) * 100).astype(int),
            0
        )
    porcentaje = np.clip(porcentaje, 0, 100)

    return asignacion, costo_total, potencia_kw, porcentaje


# =====================================================================================
# EVALUACIÓN DE APTITUD (Función Objetivo + Penalización)
# =====================================================================================
def evaluar_aptitud(poblacion: np.ndarray, demanda: float, temperatura: float = 0.0):
    """
    Evaluador de la Función Objetivo con Penalización Asimétrica y Costo Térmico.

    ──────────────────────────────────────────────────────────────────
    MODELO DE COSTO CUADRÁTICO-TÉRMICO:
    ──────────────────────────────────────────────────────────────────
    La temperatura exterior T degrada la eficiencia de cada generador j
    de forma distinta según su coeficiente de sensibilidad térmica αⱼ.

    Función de costo extendida:
        Cⱼ(Pⱼ, T) = base_j · Pⱼ  +  αⱼ · (T/100) · Pⱼ²
                    └─ lineal ─┘   └──── cuadrático-térmico ─────┘

    Costo total del cromosoma x:
        f(x, T) = Σⱼ [ base_j · Pⱼ  +  αⱼ · (T/100) · Pⱼ² ]

    POR QUÉ EL AG ES NECESARIO CON ESTE MODELO:
        - El término αⱼ·(T/100)·Pⱼ² rompe la separabilidad lineal del problema.
        - La solución óptima puede requerir repartir carga entre generadores.
        - El Voraz ordena por base_j pero ignora que el costo marginal real
          crece con la carga: dC/dP = base_j + 2αⱼ(T/100)Pⱼ.
        - El AG evalúa el costo TOTAL del cromosoma → puede descubrir
          repartos óptimos que el Voraz descarta.

    FORMULACIÓN COMPLETA CON PENALIZACIÓN:
        Minimizar:  f(x,T) + P(x)
        P(x) = 0                       si Σ Pⱼ ≥ Demanda
        P(x) = 1×10⁶ + déficit×1000   si Σ Pⱼ < Demanda
    ──────────────────────────────────────────────────────────────────

    Parámetros
    ----------
    poblacion   : ndarray (N, 8) — N cromosomas, cada gen ∈ {0..100} (% carga)
    demanda     : float          — demanda en kW (salida del SID Mamdani)
    temperatura : float          — temperatura exterior en °C [0..100]

    Retorna
    -------
    aptitud      : ndarray (N,)   — f(x,T) + P(x) por cromosoma (a minimizar)
    costos       : ndarray (N,)   — costo limpio con término térmico (sin penalización)
    potencia_tot : ndarray (N,)   — potencia total entregada por cromosoma
    kw_por_gen   : ndarray (N, 8) — kW asignados por generador por cromosoma
    """
    # 1. Alelos enteros [0,100] → fracción decimal [0.0, 1.0]
    fraccion = poblacion / 100.0                                          # (N, 8)

    # 2. Potencia aportada por generador: Pⱼ = fracción_j × Capacidad_j
    kw_por_gen = fraccion * GENERADORES[:, 0]                            # (N, 8)

    # 3. Costo lineal base: Σⱼ(Pⱼ × base_j)
    costos_lineales = np.sum(kw_por_gen * GENERADORES[:, 1], axis=1)    # (N,)

    # 4. Penalización térmica cuadrática: Σⱼ(αⱼ × (T/100) × Pⱼ²)
    #    Broadcasting: COEFS_TERMICOS (8,) × kw²(N,8) → suma por fila → (N,)
    t_normalizada   = temperatura / 100.0
    costos_termicos = np.sum(
        COEFICIENTES_TERMICOS * t_normalizada * kw_por_gen**2, axis=1
    )                                                                    # (N,)

    # 5. Costo total con modelo térmico
    costos = costos_lineales + costos_termicos                           # (N,)

    # 6. Potencia total entregada por cromosoma
    potencia_tot = np.sum(kw_por_gen, axis=1)                           # (N,)

    # 7. Penalización asimétrica exterior vectorizada (sin ciclos for)
    #    ① Salto de barrera 1e6 → separa infactibles de factibles
    #    ② Componente déficit×1000 → ordena infactibles por gravedad
    deficit  = demanda - potencia_tot
    penalizacion = np.where(deficit > 0, 1e6 + deficit * 1000.0, 0.0)

    aptitud = costos + penalizacion                                      # (N,)

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
    Algoritmo Genético Vectorizado — NumPy puro (sin librerías de caja negra).

    ──────────────────────────────────────────────────────────────────────────────
    FUNDAMENTO TEÓRICO (Holland, 1992; Goldberg, 1989):
    ──────────────────────────────────────────────────────────────────────────────
    Basado en el Teorema de los Esquemas (Holland, 1975):
        El AG procesa implícitamente O(N³) bloques de construcción (esquemas)
        con solo N evaluaciones por generación («paralelismo implícito»).
        Esquemas con aptitud superior al promedio reciben crecimiento exponencial
        en generaciones sucesivas → convergencia al óptimo con alta probabilidad.

    Parámetros estándar empleados:
        Pc = 0.85  → dentro del rango teórico [0.6, 0.9] (Goldberg, 1989)
        μ  = 0.10  → calibrado para espacio discreto 101⁸
        k  = 3     → presión selectiva moderada sin convergencia prematura
        Élites = 2 → garantiza aptitud no creciente: f*(t+1) ≤ f*(t)

    Cromosoma:  x = [g₁,g₂,g₃,g₄,g₅,g₆,g₇,g₈] ∈ {0,...,100}⁸
    Espacio de búsqueda:  |S| = 101⁸ ≈ 1.08×10¹⁶  (justifica metaheurística)
    ──────────────────────────────────────────────────────────────────────────────

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
    mejor_cromosoma   : ndarray (8,)  — cromosoma óptimo (% de carga por gen.)
    mejor_asignacion  : ndarray (8,)  — kW despachados por generador
    mejor_costo       : float         — costo USD con modelo térmico (sin penalización)
    mejor_potencia    : float         — potencia total despachada en kW
    historial_aptitud : list[float]   — aptitud mínima por generación (con penalización)
    historial_costo   : list[float]   — costo mínimo limpio por generación (NaN si infactible)
    gen_parada        : int           — generación real en que el AG se detuvo
    """
    # ── INICIALIZACIÓN (t=0): Población aleatoria uniforme ───────────────────
    # P₀ ~ U({0,...,100}⁸)^N — cobertura uniforme del hiperespacio de búsqueda
    poblacion = np.random.randint(0, 101, size=(tam_poblacion, N_GENERADORES))

    historial_aptitud: list = []   # aptitud mínima(t) — incluye penalización
    historial_costo:   list = []   # costo mínimo limpio(t) — NaN si sin solución válida

    # ── Criterio de parada por estancamiento ─────────────────────────────────
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
        # Ordenamos ascendente (menor aptitud = individuo más apto)
        indices_ordenados = np.argsort(aptitud)
        elites            = poblacion[indices_ordenados[:num_elites]].copy()

        # ── REGISTRO DE HISTORIALES ───────────────────────────────────────────
        aptitud_actual = float(aptitud[indices_ordenados[0]])
        historial_aptitud.append(aptitud_actual)

        mascara_validos = potencia_tot >= demanda
        if np.any(mascara_validos):
            historial_costo.append(float(np.min(costos[mascara_validos])))
        else:
            historial_costo.append(float('nan'))   # generación completamente infactible

        # ── DETECCIÓN DE CONVERGENCIA (Criterio de Estancamiento) ────────────
        if aptitud_actual < mejor_aptitud_global - 1e-6:   # mejora significativa
            mejor_aptitud_global = aptitud_actual
            estancamiento = 0
        else:
            estancamiento += 1

        if estancamiento >= PACIENCIA:
            gen_parada = gen + 1       # generación real de convergencia
            break

        # ── FASE B: SELECCIÓN POR TORNEO ESTRICTO (k=3) ──────────────────────
        # ┌────────────────────────────────────────────────────────────────────┐
        # │  Para cada posición i de la nueva población:                      │
        # │    1. Elegir 3 índices al azar (sin reemplazo)                    │
        # │    2. El cromosoma con menor aptitud gana el torneo               │
        # │    3. El ganador ocupa la posición i en la nueva población         │
        # │  Presión selectiva k=3 → intermedia entre k=2 y k=5              │
        # └────────────────────────────────────────────────────────────────────┘
        nueva_poblacion = np.zeros_like(poblacion)
        for i in range(tam_poblacion):
            combatientes = np.random.choice(tam_poblacion, size=3, replace=False)
            ganador      = combatientes[np.argmin(aptitud[combatientes])]
            nueva_poblacion[i] = poblacion[ganador]

        # ── FASE C: CRUZAMIENTO DE UN PUNTO (Pc = 0.85) ──────────────────────
        # ┌────────────────────────────────────────────────────────────────────┐
        # │  Sea p ∈ {1,...,7} el punto de corte (aleatorio).                 │
        # │                                                                    │
        # │  Padre₁ = [ G₁  G₂ ··· Gₚ | G_{p+1} ··· G₈ ]                   │
        # │  Padre₂ = [ H₁  H₂ ··· Hₚ | H_{p+1} ··· H₈ ]                   │
        # │                              ↕ intercambio de colas               │
        # │  Hijo₁  = [ G₁  G₂ ··· Gₚ    H_{p+1} ··· H₈ ]                  │
        # │  Hijo₂  = [ H₁  H₂ ··· Hₚ    G_{p+1} ··· G₈ ]                  │
        # └────────────────────────────────────────────────────────────────────┘
        for i in range(num_elites, tam_poblacion - 1, 2):   # élites inmunes
            if np.random.rand() < 0.85:
                punto_corte = np.random.randint(1, N_GENERADORES)
                padre1 = nueva_poblacion[i].copy()
                padre2 = nueva_poblacion[i + 1].copy()
                nueva_poblacion[i,     punto_corte:] = padre2[punto_corte:]
                nueva_poblacion[i + 1, punto_corte:] = padre1[punto_corte:]

        # ── FASE D: MUTACIÓN UNIFORME VECTORIZADA ────────────────────────────
        # ┌────────────────────────────────────────────────────────────────────┐
        # │  Máscara booleana: M = (D < μ),  D ~ U[0,1]^{N×8}               │
        # │  Si M[i,j] = True  → gen mutado  = R[i,j] ~ U({0,...,100})       │
        # │  Si M[i,j] = False → gen intacto = nueva_pop[i,j]                │
        # │  Élites B[:num_elites,:] = False  →  INMUNES a mutación           │
        # └────────────────────────────────────────────────────────────────────┘
        mascara_mutacion   = np.random.rand(tam_poblacion, N_GENERADORES) < tasa_mutacion
        mascara_mutacion[:num_elites] = False                  # élites no mutan
        alelos_aleatorios  = np.random.randint(0, 101, size=(tam_poblacion, N_GENERADORES))
        nueva_poblacion    = np.where(mascara_mutacion, alelos_aleatorios, nueva_poblacion)

        # ── INYECCIÓN DE ÉLITES: Garantiza aptitud no creciente ──────────────
        # Los num_elites mejores de t se copian directamente a t+1.
        # Propiedad: f*(t+1) ≤ f*(t) para todo t  (Whitley, 1989).
        nueva_poblacion[:num_elites] = elites

        poblacion = nueva_poblacion

    # ── POST-CONVERGENCIA: Extraer el cromosoma con menor aptitud ────────────
    aptitud_final, costos_final, potencia_final, asignacion_final = \
        evaluar_aptitud(poblacion, demanda, temperatura)

    indice_mejor = np.argmin(aptitud_final)

    return (
        poblacion[indice_mejor],             # mejor_cromosoma
        asignacion_final[indice_mejor],      # mejor_asignacion (kW por gen.)
        costos_final[indice_mejor],          # mejor_costo (USD, sin penalización)
        potencia_final[indice_mejor],        # mejor_potencia (kW total)
        historial_aptitud,                   # historial de aptitud mínima
        historial_costo,                     # historial de costo limpio mínimo
        gen_parada,                          # generación real de parada
    )
