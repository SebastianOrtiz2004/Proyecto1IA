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
GENERATORS = np.array([
    [300.0, 150.0],   # Gen 1: Cap. media,      costo alto
    [500.0, 100.0],   # Gen 2: Cap. alta,        costo moderado
    [200.0, 200.0],   # Gen 3: Cap. baja,        BACKUP CRÍTICO (costoso)
    [400.0,  80.0],   # Gen 4: Cap. grande,      base de carga eficiente
    [600.0,  90.0],   # Gen 5: Cap. MÁXIMA,      muy eficiente
    [150.0, 250.0],   # Gen 6: Cap. mínima,      EMERGENCIA (el más caro)
    [450.0, 110.0],   # Gen 7: Cap. media-alta,  costo moderado
    [350.0,  70.0],   # Gen 8: Cap. media,       el MÁS EFICIENTE del parque
])

# Dimensión del cromosoma (número de genes = número de generadores)
N_GENES: int = len(GENERATORS)   # = 8


# =====================================================================================
# FUNCIÓN DE REFERENCIA: DESPACHO GREEDY ÓPTIMO
# =====================================================================================
def greedy_dispatch(demand: float):
    """
    Algoritmo Greedy de Despacho Económico — Solución de Referencia (Benchmark).

    ──────────────────────────────────────────────────────────────────────────────
    TEOREMA DE OPTIMALIDAD (Principio de Optimalidad de Bellman):
    ──────────────────────────────────────────────────────────────────────────────
    Para una función de costo lineal separable de la forma:
        f(x) = Σ_{j} cᵢ · xⱼ  (cᵢ constante, xⱼ ≥ 0)
    y una restricción de suma Σxⱼ ≥ D, la política óptima es:
        → Asignar carga máxima al generador de menor costo unitario
        → Continuar con el siguiente más barato hasta cubrir D

    Esta es la solución ÓPTIMA GLOBAL garantizada para este tipo de problema.
    Sirve como benchmark para evaluar la calidad de la metaheurística GA.

    NOTA ACADÉMICA:
    Si el modelo de costo fuera NO LINEAL (ej. cuadrático: aⱼ·Pⱼ² + bⱼ·Pⱼ + cⱼ),
    el greedy deja de ser óptimo y el GA se justifica como necesidad, no solo
    como alternativa. Esta es la motivación real de los métodos metaheurísticos
    en problemas de Unit Commitment reales.
    ──────────────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    demand : float — demanda en kW a satisfacer

    Returns
    -------
    allocation : ndarray (8,) — kW asignados por generador
    cost       : float        — costo total mínimo USD (óptimo global)
    total_kw   : float        — potencia total despachada
    load_pct   : ndarray (8,) — porcentaje de carga por generador [0..100]
    """
    # Ordenar generadores por costo ascendente (el más barato primero)
    priority_order = np.argsort(GENERATORS[:, 1])     # índices: [7,3,4,1,6,0,2,5]

    allocation = np.zeros(N_GENES)
    remaining  = demand

    for idx in priority_order:
        if remaining <= 0:
            break
        # Asignar mínimo entre lo que queda por cubrir y la capacidad máxima del gen.
        assigned_kw      = min(remaining, GENERATORS[idx, 0])
        allocation[idx]  = assigned_kw
        remaining       -= assigned_kw

    total_kw = float(np.sum(allocation))
    cost     = float(np.sum(allocation * GENERATORS[:, 1]))

    # Convertir kW a % de carga (redondeado a entero, como el cromosoma GA)
    with np.errstate(divide='ignore', invalid='ignore'):
        load_pct = np.where(
            GENERATORS[:, 0] > 0,
            np.round((allocation / GENERATORS[:, 0]) * 100).astype(int),
            0
        )
    load_pct = np.clip(load_pct, 0, 100)

    return allocation, cost, total_kw, load_pct


# =====================================================================================
# EVALUACIÓN DE FITNESS (Función Objetivo + Penalización)
# =====================================================================================
def evaluate_fitness(population: np.ndarray, demand: float):
    """
    Evaluador de la Función Objetivo con Penalización Asimétrica.

    ──────────────────────────────────────────────────────────────────
    FORMULACIÓN MATEMÁTICA (Investigación de Operaciones):
    ──────────────────────────────────────────────────────────────────
    Minimizar:   f(x) = Σ_{j=1}^{8} [ frac_j · Cap_j · Costo_j ] + P(x)

    Restricción dura (g):
        g(x): Σ_{j=1}^{8} [ frac_j · Cap_j ] ≥ Demanda

    Función de Penalización Asimétrica (Exterior Penalty Method):
        P(x) = 0                          si g(x) se cumple
        P(x) = 1×10⁶ + déficit×1000      si g(x) se viola

    Propiedades de la penalización:
        ① max(P) ≈ 1e6 + 2950×1000 ≈ 3.95×10⁶  >>  max(f_real) ≈ 737,500
        → Toda solución infactible es SIEMPRE peor que cualquier solución factible.
        ② La componente 1e6 es el "salto de barrera" que separa los espacios.
        ③ El término déficit×1000 ordena infactibles por gravedad del déficit.
    ──────────────────────────────────────────────────────────────────

    Parameters
    ----------
    population : ndarray (N, 8) — N cromosomas, cada gen ∈ {0..100} (% carga)
    demand     : float          — demanda en kW (salida del FIS Mamdani)

    Returns
    -------
    fitness   : ndarray (N,)   — valor f(x) por cromosoma (a minimizar)
    costs     : ndarray (N,)   — costo operativo LIMPIO (sin penalización)
    total_kw  : ndarray (N,)   — potencia total entregada por cromosoma
    kw_gen    : ndarray (N, 8) — kW asignados por generador por cromosoma
    """
    # 1. Alelos enteros [0,100] → fracción decimal [0.0, 1.0]
    frac = population / 100.0                               # (N, 8)

    # 2. Potencia aportada: Broadcasting implícito (N,8) × (8,) → (N,8)
    kw_gen = frac * GENERATORS[:, 0]                        # (N, 8)

    # 3. Costo operativo: Σ_j(kW_j × USD/kW_j) por cromosoma
    costs = np.sum(kw_gen * GENERATORS[:, 1], axis=1)       # (N,)

    # 4. Potencia total entregada por cromosoma
    total_kw = np.sum(kw_gen, axis=1)                       # (N,)

    # 5. Penalización asimétrica vectorizada — sin ciclos for
    deficit = demand - total_kw
    penalty = np.where(deficit > 0, 1e6 + deficit * 1000.0, 0.0)

    fitness = costs + penalty                                # (N,)

    return fitness, costs, total_kw, kw_gen


# =====================================================================================
# ALGORITMO GENÉTICO PRINCIPAL
# =====================================================================================
def run_genetic_algorithm(
    demand:        float,
    pop_size:      int   = 60,
    generations:   int   = 100,
    mutation_rate: float = 0.10,
    elite_k:       int   = 2
):
    """
    Algoritmo Genético Vectorizado — NumPy Puro (sin librerías de caja negra).

    ──────────────────────────────────────────────────────────────────────────────
    FUNDAMENTO TEÓRICO (Holland, 1992; Goldberg, 1989):
    ──────────────────────────────────────────────────────────────────────────────
    Basado en el Teorema de los Schemata (Holland, 1975):
        El GA procesa implícitamente O(N³) bloques de construcción (schemata)
        con solo N evaluaciones por generación ("paralelismo implícito").
        Schemata con fitness superior al promedio reciben crecimiento exponencial
        en generaciones sucesivas → convergencia al óptimo con alta probabilidad.

    Parámetros estándar empleados:
        Pc = 0.85  → dentro del rango teórico [0.6, 0.9] (Goldberg, 1989)
        μ  = 0.10  → calibrado para espacio discreto 101⁸; mayor que 1/L=0.125
                     (recomendación teórica para GAs continuos) a fin de mantener
                     diversidad en espacio de búsqueda de alta dimensionalidad.
        k  = 3     → torneo con k=3 provee presión selectiva moderada sin
                     convergencia prematura (De Jong & Sarma, 1993)
        Elitismo k=2 → garantiza monotonía no-creciente del mejor fitness (Whitley, 1989)

    NOTA sobre el modelo de costo:
        Se emplea función de costo lineal f(x)=Σcᵢ·Pᵢ para validar el operador
        genético. El GA está diseñado para aceptar cualquier f(x) diferenciable
        o no, incluyendo modelos cuadráticos de función de calor no separables,
        sin modificar ningún operador — esta es su ventaja sobre métodos exactos.
    ──────────────────────────────────────────────────────────────────────────────

    Cromosoma:  x = [g₁,g₂,g₃,g₄,g₅,g₆,g₇,g₈] ∈ {0,...,100}⁸
    Espacio de búsqueda:  |S| = 101⁸ ≈ 1.08×10¹⁶  (justifica metaheurística)

    Parameters
    ----------
    demand        : float — demanda en kW (del FIS Mamdani)
    pop_size      : int   — tamaño de población N
    generations   : int   — máximo de generaciones t_max
    mutation_rate : float — probabilidad de mutación por gen μ ∈ (0,1)
    elite_k       : int   — individuos élite preservados (elitismo)

    Returns
    -------
    best_chromosome : ndarray (8,)  — cromosoma óptimo (% de carga)
    best_allocation : ndarray (8,)  — kW por generador
    best_cost       : float         — costo USD (sin penalización)
    best_kw         : float         — potencia total kW
    fitness_history : list[float]   — min fitness(t) con penalización
    cost_history    : list[float]   — min costo limpio(t); NaN si gen. infactible
    gen_stopped     : int           — generación real en que el GA se detuvo
    """
    # ── INICIALIZACIÓN (t=0): Población aleatoria uniforme ──────────────────────
    # P₀ ~ U({0,...,100}⁸)^N — cobertura uniforme del hiperespacio
    population = np.random.randint(0, 101, size=(pop_size, N_GENES))

    fitness_history: list = []   # min fitness(t) — incluye penalización
    cost_history:    list = []   # min costo limpio(t) — NaN si sin solución válida

    # ── Criterio de parada por estagnación ──────────────────────────────────────
    # Si el fitness mínimo no mejora en PATIENCE generaciones consecutivas,
    # el GA ha convergido: continuar es computacionalmente inútil.
    # PATIENCE = max(20, 20% del total) — adaptativo al parámetro del usuario.
    PATIENCE         = max(20, generations // 5)
    stagnation       = 0
    best_fitness_ever = float('inf')
    gen_stopped      = generations   # se actualizará si hay parada anticipada

    for gen in range(generations):

        # ── FASE A: EVALUACIÓN FENOTÍPICA ────────────────────────────────────
        fitness, costs, total_kw, _ = evaluate_fitness(population, demand)

        # ── ELITISMO: Preservar top-k ANTES de modificar la población ────────
        # Ordenamos ascendente (menor fitness = individuo más apto)
        sorted_idx = np.argsort(fitness)
        elites     = population[sorted_idx[:elite_k]].copy()   # copia profunda

        # ── REGISTRO DE HISTORIALES (AMBOS — siempre antes del break) ────────
        current_best = float(fitness[sorted_idx[0]])
        fitness_history.append(current_best)

        valid_mask = total_kw >= demand
        if np.any(valid_mask):
            cost_history.append(float(np.min(costs[valid_mask])))
        else:
            cost_history.append(float('nan'))   # generación completamente infactible

        # ── DETECCIÓN DE CONVERGENCIA (Criterio de Estagnación) ──────────────
        if current_best < best_fitness_ever - 1e-6:   # mejora significativa
            best_fitness_ever = current_best
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= PATIENCE:
            gen_stopped = gen + 1       # generación real de convergencia
            break

        # ── FASE B: SELECCIÓN POR TORNEO ESTRICTO (k=3) ──────────────────────
        # ┌────────────────────────────────────────────────────────────────────┐
        # │  PIZARRA: Para cada posición i de la nueva población:             │
        # │    1. Elegir 3 índices al azar del pool (sin reemplazo)           │
        # │    2. El individuo con menor fitness gana el torneo               │
        # │    3. El ganador ocupa la posición i en new_population            │
        # │  Presión selectiva k=3 → intermedia entre k=2 (baja) y k=5 (alta)│
        # │  Referencia: De Jong & Sarma (1993), Tournament Selection         │
        # └────────────────────────────────────────────────────────────────────┘
        new_population = np.zeros_like(population)
        for i in range(pop_size):
            combatants = np.random.choice(pop_size, size=3, replace=False)
            winner     = combatants[np.argmin(fitness[combatants])]
            new_population[i] = population[winner]

        # ── FASE C: CRUZAMIENTO DE UN PUNTO (Pc = 0.85) ──────────────────────
        # ┌────────────────────────────────────────────────────────────────────┐
        # │  PIZARRA: Sea p ∈ {1,...,7} el punto de corte (aleatorio).        │
        # │                                                                    │
        # │  Padre₁ = [ G₁  G₂ ··· Gₚ | G_{p+1} ··· G₈ ]                   │
        # │  Padre₂ = [ H₁  H₂ ··· Hₚ | H_{p+1} ··· H₈ ]                   │
        # │                                ↕ intercambio de colas             │
        # │  Hijo₁  = [ G₁  G₂ ··· Gₚ    H_{p+1} ··· H₈ ]  (slicing O(1))  │
        # │  Hijo₂  = [ H₁  H₂ ··· Hₚ    G_{p+1} ··· G₈ ]                  │
        # │                                                                    │
        # │  Pc=0.85 dentro del rango teórico [0.6,0.9] (Goldberg, 1989)     │
        # └────────────────────────────────────────────────────────────────────┘
        for i in range(elite_k, pop_size - 1, 2):      # élites en [0..elite_k) — inmunes
            if np.random.rand() < 0.85:
                p  = np.random.randint(1, N_GENES)     # punto de corte ∈ {1,...,7}
                p1 = new_population[i].copy()
                p2 = new_population[i + 1].copy()
                new_population[i, p:]     = p2[p:]     # Hijo₁ hereda cola de Padre₂
                new_population[i + 1, p:] = p1[p:]    # Hijo₂ hereda cola de Padre₁

        # ── FASE D: MUTACIÓN UNIFORME VECTORIZADA ────────────────────────────
        # ┌────────────────────────────────────────────────────────────────────┐
        # │  PIZARRA: Sea D ∈ M_{N×8}(U[0,1]) — matriz de ruido uniforme.    │
        # │  Máscara booleana: B = (D < μ),  con μ = mutation_rate.           │
        # │                                                                    │
        # │  Si B[i,j] = True  → gen mutado  = R[i,j] ~ U({0,...,100})       │
        # │  Si B[i,j] = False → gen intacto = new_pop[i,j]                  │
        # │                                                                    │
        # │  Implementación vectorizada:                                       │
        # │      x_new = np.where(B, R, x_old)   ← O(N·L), sin ciclos for   │
        # │                                                                    │
        # │  E[mutaciones/cromosoma] = L × μ = 8 × 0.10 = 0.8               │
        # │  (vs. recomendación teórica 1/L=0.125 para GAs continuos;        │
        # │   se usa μ mayor para mantener diversidad en espacio discreto)    │
        # │                                                                    │
        # │  Los élites B[:elite_k,:] = False  →  INMUNES a mutación         │
        # └────────────────────────────────────────────────────────────────────┘
        mut_mask       = np.random.rand(pop_size, N_GENES) < mutation_rate
        mut_mask[:elite_k] = False                      # élites no mutan
        random_alleles = np.random.randint(0, 101, size=(pop_size, N_GENES))
        new_population = np.where(mut_mask, random_alleles, new_population)

        # ── INYECCIÓN DE ÉLITES: Garantiza monotonía no-creciente ────────────
        # Los elite_k mejores de t se copian directamente a t+1.
        # Propiedad: f*(t+1) ≤ f*(t) para todo t  (Whitley, 1989).
        new_population[:elite_k] = elites

        population = new_population

    # ── POST-CONVERGENCIA: Extraer el cromosoma óptimo ──────────────────────────
    final_fitness, final_costs, final_total_kw, final_kw_alloc = \
        evaluate_fitness(population, demand)

    best_idx = np.argmin(final_fitness)

    return (
        population[best_idx],          # best_chromosome
        final_kw_alloc[best_idx],      # best_allocation (kW por gen.)
        final_costs[best_idx],         # best_cost (USD, sin penalización)
        final_total_kw[best_idx],      # best_kw  (potencia total)
        fitness_history,               # historia fitness mínimo con penalización
        cost_history,                  # historia costo limpio mínimo
        gen_stopped,                   # generación real de parada
    )
