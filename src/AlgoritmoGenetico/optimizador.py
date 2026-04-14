"""
AlgoritmoGenetico/optimizador.py
Orquestador del Algoritmo Genético Modular.
Gestiona el ciclo de vida evolutivo: Selección -> Cruzamiento -> Mutación -> Elitismo.
"""

import random
from .config import GENERADORES, N_GENERADORES, COEFICIENTES_TERMICOS
from .evaluacion import evaluar_poblacion
from .seleccion import seleccion_torneo
from .cruzamiento import aplicar_cruzamiento
from .mutacion import aplicar_mutacion

def despacho_voraz(demanda: float, temperatura: float):
    """Benchmark voraz usando el mismo modelo de costo del AG."""
    orden = sorted(range(N_GENERADORES), key=lambda i: GENERADORES[i][1])
    asignacion = [0.0] * N_GENERADORES
    restante = max(0.0, float(demanda))
    for i in orden:
        if restante <= 0:
            break
        kw = min(restante, GENERADORES[i][0])
        asignacion[i] = kw
        restante -= kw

    tn = max(0.0, float(temperatura)) / 100.0
    costo_lineal = sum(asignacion[i] * GENERADORES[i][1] for i in range(N_GENERADORES))
    costo_termico = sum(COEFICIENTES_TERMICOS[i] * tn * (asignacion[i] ** 2) for i in range(N_GENERADORES))
    costo = costo_lineal + costo_termico
    porcentajes = [int(round((asignacion[i]/GENERADORES[i][0])*100)) if GENERADORES[i][0]>0 else 0 for i in range(N_GENERADORES)]
    return asignacion, costo, sum(asignacion), porcentajes

def ejecutar_ag(demanda: float, temperatura: float, tam_poblacion=60, generaciones=100, tasa_mutacion=0.1, num_elites=2, modo_rafaga=False):
    """Bucle evolutivo principal."""
    tam_poblacion = max(2, int(tam_poblacion))
    generaciones = max(1, int(generaciones))
    tasa_mutacion = min(1.0, max(0.0, float(tasa_mutacion)))
    num_elites = max(1, min(int(num_elites), tam_poblacion - 1))

    if modo_rafaga:
        tam_poblacion, generaciones = min(tam_poblacion, 35), min(generaciones, 45)

    # 1. Población Inicial
    poblacion = [[random.randint(0, 100) for _ in range(N_GENERADORES)] for _ in range(tam_poblacion)]
    
    hist_apt, hist_costo = [], []
    mejor_apt_global = float('inf')
    estancamiento, paciencia = 0, max(20, generaciones // 5)
    gen_parada = generaciones

    for gen in range(generaciones):
        # 2. EVALUACIÓN
        apt, costos, pots, _ = evaluar_poblacion(poblacion, demanda, temperatura)
        
        # ELITISMO (Guardar mejores)
        idx_ord = sorted(range(tam_poblacion), key=lambda i: apt[i])
        elites = [poblacion[i][:] for i in idx_ord[:num_elites]]
        
        apt_actual = apt[idx_ord[0]]
        hist_apt.append(apt_actual)
        validos = [i for i in range(tam_poblacion) if pots[i] >= demanda]
        hist_costo.append(min(costos[i] for i in validos) if validos else float('nan'))

        # PARADA ANTICIPADA
        if apt_actual < mejor_apt_global - 1e-6:
            mejor_apt_global = apt_actual; estancamiento = 0
        else:
            estancamiento += 1
        if estancamiento >= paciencia:
            gen_parada = gen + 1; break

        # 3. SELECCIÓN
        poblacion = seleccion_torneo(poblacion, apt, k=3)
        
        # 4. CRUZAMIENTO
        poblacion[num_elites:] = aplicar_cruzamiento(poblacion[num_elites:], pc=0.85)
        
        # 5. MUTACIÓN
        poblacion[num_elites:] = aplicar_mutacion(poblacion[num_elites:], mu=tasa_mutacion)
        
        # INYECCIÓN ELITES
        poblacion[:num_elites] = elites

    # RESULTADO FINAL
    apt_f, cost_f, pot_f, kw_f = evaluar_poblacion(poblacion, demanda, temperatura)
    idx_b = min(range(tam_poblacion), key=lambda i: apt_f[i])
    
    return poblacion[idx_b], kw_f[idx_b], cost_f[idx_b], pot_f[idx_b], hist_apt, hist_costo, gen_parada
