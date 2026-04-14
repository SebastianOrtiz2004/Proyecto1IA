"""
AlgoritmoGenetico/seleccion.py
Implementa la presión selectiva mediante torneos.
"""
import random

def seleccion_torneo(poblacion: list, aptitud: list, k: int = 3):
    """
    Selección por torneos de tamaño k. 
    Elegimos k individuos al azar y el más apto (menor aptitud) gana.
    """
    nueva_poblacion = []
    tam_poblacion = len(poblacion)
    
    for _ in range(tam_poblacion):
        combatientes = random.sample(range(tam_poblacion), k)
        ganador = min(combatientes, key=lambda i: aptitud[i])
        nueva_poblacion.append(poblacion[ganador][:])
        
    return nueva_poblacion
