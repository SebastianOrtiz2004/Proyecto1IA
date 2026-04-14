"""
LogicaDifusa/inferencia.py (FASE 2)
Aplica la T-Norma (Mínimo) sobre los antecedentes por cada regla minada.
"""

def aplicar_inferencia(mu_prod: dict, mu_temp: dict, reglas: list):
    """
    Motor de implicación Mamdani. 
    Activa cada regla usando el operador Mínimo:
    activacion = min(mu_produccion, mu_temperatura)
    """
    activaciones = []
    for conj_prod, conj_temp, conj_dem in reglas:
        # Se asume conjuncion AND mediante T-Norma (Mínimo)
        grado_activacion = min(mu_prod[conj_prod], mu_temp[conj_temp])
        
        if grado_activacion > 0:
            activaciones.append((grado_activacion, conj_dem))
            
    return activaciones
