
import random
from .config import N_GENERADORES

# Cruce de un punto: mantiene bloques de genes de cada padre.
def aplicar_cruzamiento(poblacion: list, pc: float = 0.85):

    p_len = len(poblacion)
    # Recombinar de 2 en 2
    for i in range(0, p_len - 1, 2):
        if random.random() < pc:
            punto = random.randint(1, N_GENERADORES - 1)
            # Intercambio de segmentos
            padre1 = poblacion[i][:]
            padre2 = poblacion[i + 1][:]
            poblacion[i][punto:]     = padre2[punto:]
            poblacion[i + 1][punto:] = padre1[punto:]
            
    return poblacion
