"""
AlgoritmoGenetico/mutacion.py
Operador de mutación uniforme para mantener la diversidad.
"""
import random

def aplicar_mutacion(poblacion: list, mu: float = 0.1):
    """
    Mutación uniforme aleatoria por gen. 
    Evita la convergencia prematura a óptimos locales.
    """
    tam_poblacion = len(poblacion)
    if tam_poblacion == 0: return poblacion
    n_genes = len(poblacion[0])
    
    for i in range(tam_poblacion):
        for j in range(n_genes):
            if random.random() < mu:
                # Se asigna un nuevo valor aleatorio entre 0 y 100
                poblacion[i][j] = random.randint(0, 100)
                
    return poblacion
