"""
AlgoritmoGenetico/evaluacion.py
Función de Aptitud (Fitness) con penalización asimétrica.
"""
from .config import GENERADORES, COEFICIENTES_TERMICOS, N_GENERADORES

def evaluar_poblacion(poblacion: list, demanda: float, temperatura: float):
    """
    Evalúa cada cromosoma. 
    Fitness = Costo(lineal + térmico) + Penalización(déficit)
    """
    tn = temperatura / 100.0
    aptitudes = []
    costos = []
    potencias = []
    kw_asignados = []

    for cromosoma in poblacion:
        # Porcentaje -> kW
        kw = [(cromosoma[i] / 100.0) * GENERADORES[i][0] for i in range(N_GENERADORES)]
        
        # Costo base + cuadrático térmico
        c_lineal = sum(kw[i] * GENERADORES[i][1] for i in range(N_GENERADORES))
        c_termico = sum(COEFICIENTES_TERMICOS[i] * tn * (kw[i]**2) for i in range(N_GENERADORES))
        costo_total = c_lineal + c_termico
        
        potencia_gen = sum(kw)
        deficit = demanda - potencia_gen
        
        # Penalización exterior asimétrica
        penalizacion = (1e6 + deficit * 1000.0) if deficit > 0 else 0.0
        
        aptitudes.append(costo_total + penalizacion)
        costos.append(costo_total)
        potencias.append(potencia_gen)
        kw_asignados.append(kw)
        
    return aptitudes, costos, potencias, kw_asignados
