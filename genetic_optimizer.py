import numpy as np

# =====================================================================================
# MATRIZ DEL MODELO DE OPERACIONES: FLOTA DE GENERADORES DIÉSEL (Modo Isla)
# =====================================================================================
# La matriz mapea los 4 Generadores del problema.
# Columna 1: Capacidad Máxima Nominal [kW]
# Columna 2: Costo Operativo Directo [USD / kW]
GENERATORS = np.array([
    [300.0, 150.0],  # Gen 1: Capacidad media, costo muy alto
    [500.0, 100.0],  # Gen 2: Capacidad alta, costo medio
    [200.0, 200.0],  # Gen 3: Capacidad baja, costo brutalmente caro (Backup crítico)
    [400.0,  80.0]   # Gen 4: Capacidad grande, costo altamente eficiente (Base de carga)
])

def evaluate_fitness(population, demand):
    """
    Evaluador de la Función Objetivo y Función de Penalización de Investigaciones de Operaciones.
    poblacion: matiz (N x 4) donde la fila 'i' es un cromosoma y celda (i, j) el Gen (%)
    
    Justificación Matemática:
    Minimizar f(x) = \sum_{j=1}^{4} (Porcentaje_j * Cap_j * Costo_j) + P(x)
    Restricción Sujeta a: \sum_{j=1}^{4} (Porcentaje_j * Cap_j) >= Demanda
    """
    # 1. Transformación Lineal: Convertir alelos enteros (0 a 100) a fracción decimal (0 a 1.0)
    fractional_load = population / 100.0
    
    # 2. Despacho Asignado (kW aportado por cada generador del cromosoma)
    # Multiplicación Element-Wise con broadcasting matricial
    kw_generated = fractional_load * GENERATORS[:, 0]
    
    # 3. Costo Directo por cromosoma = Sumatoria Aritmética de kW * Costo_USD
    costs = np.sum(kw_generated * GENERATORS[:, 1], axis=1)
    
    # 4. Potencia total entregada por cada individuo
    total_kw = np.sum(kw_generated, axis=1)
    
    # 5. RESTRICCIÓN MATEMÁTICA DURA (Penalización Asimétrica P(x))
    # Si la potencia entregada < demanda (falla estructural grave), el fitness se penaliza 
    # abruptamente a un valor hiper-positivo (+ de 1 Millon) para forzar selección negativa.
    deficit = demand - total_kw
    
    # Vectorización NumPy: condicional if_else sin ciclos for (Alta eficiencia matemática)
    penalty = np.where(deficit > 0, 1e6 + (deficit * 1000.0), 0)
    
    # Función Fitness Final (a Minimizar obligatoriamente)
    fitness = costs + penalty
    
    return fitness, costs, total_kw, kw_generated


def run_genetic_algorithm(demand, pop_size=50, generations=100, mutation_rate=0.1):
    """
    Algoritmo Genético Vectorizado implementado rigurosamente en NumPy (Sin cajas negras de librerías externas).
    Parámetros adaptables desde la Interfaz de UI.
    """
    # INICIALIZACIÓN: Creamos el Espacio de Búsqueda Dimensional Aleatorio Uniforme Aleatorio
    # Matriz NxF (N: Individuos, F=4 Genes correspondientes a Generadores, % 0-100)
    population = np.random.randint(0, 101, size=(pop_size, 4))
    
    fitness_history = []
    
    for gen in range(generations):
        # ---------------------------------------------------------
        # FASE A: EVALUACIÓN FENOTIPICA (FITNESS)
        # ---------------------------------------------------------
        fitness, _, _, _ = evaluate_fitness(population, demand)
        
        # Registrar el mínimo global de la generación n-ésima
        best_fitness = np.min(fitness)
        fitness_history.append(best_fitness)
        
        # ---------------------------------------------------------
        # FASE B: SELECCIÓN OPERACIONAL (MÉTODO POR TORNEO ESTRICTO DE k=3)
        # ---------------------------------------------------------
        new_population = np.zeros_like(population)
        for i in range(pop_size):
            # Proceso estocástico: Elegir 3 índices al azar de la matriz y batallar
            combatants_idx = np.random.choice(pop_size, size=3, replace=False)
            best_idx = combatants_idx[np.argmin(fitness[combatants_idx])]
            new_population[i] = population[best_idx]
            
        # ---------------------------------------------------------
        # FASE C: OPERADOR DE CRUZAMIENTO O CROSSOVER LOGICO MATEMÁTICO
        # ---------------------------------------------------------
        # [COMENTARIO PARA DEFENSA EN PIZARRA]:
        # Suponiendo que seleccionamos Aleatoriamente el punto p=2 (\in {1, 2, 3})
        # Padre 1 (Row i)   = [ G1, G2 | G3, G4 ]
        # Padre 2 (Row i+1) = [ H1, H2 | H3, H4 ]
        # Hijo 1  = Recombinación [ G1, G2, H3, H4 ]
        # Hijo 2  = Recombinación [ H1, H2, G3, G4 ]
        for i in range(0, pop_size - 1, 2):
            if np.random.rand() < 0.85: # Probabilidad Prob(Cruce) = 85%.
                crossover_point = np.random.randint(1, 4)
                parent1 = new_population[i].copy()
                parent2 = new_population[i+1].copy()
                
                # Operador Slicing de Arrays para el cruce explícito
                new_population[i, crossover_point:] = parent2[crossover_point:]
                new_population[i+1, crossover_point:] = parent1[crossover_point:]
                
        # ---------------------------------------------------------
        # FASE D: OPERADOR DE MUTACIÓN TOTALMENTE ESTOCÁSTICO
        # ---------------------------------------------------------
        # [COMENTARIO PARA DEFENSA EN PIZARRA]:
        # Implementado matemáticamente calculando una Matriz Aleatoria D(Nx4) ~ U(0,1) y 
        # creando una Máscara Booleana B = (D < mutation_rate). Finalizamos con np.where.
        mask = np.random.rand(pop_size, 4) < mutation_rate
        random_alleles = np.random.randint(0, 101, size=(pop_size, 4))
        
        # Sustituimos Genes mutados mediante Álgebra Booleana Nativa
        population = np.where(mask, random_alleles, new_population)
        
    # ============================================================================
    # FASE E: PROCESAMIENTO MATRICIAL POST-CONVERGENCIA
    # ============================================================================
    final_fitness, final_costs, final_total_kw, final_kw_allocated = evaluate_fitness(population, demand)
    best_final_idx = np.argmin(final_fitness)
    
    # Extraemos el Vector de la Especie Suprema (Cromosoma Óptimo)
    best_chromosome = population[best_final_idx]
    best_cost = final_costs[best_final_idx]
    best_kw = final_total_kw[best_final_idx]
    best_allocation = final_kw_allocated[best_final_idx]
    
    return best_chromosome, best_allocation, best_cost, best_kw, fitness_history
