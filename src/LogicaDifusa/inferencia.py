
# Esta fase decide que reglas "disparan" y con que intensidad. Aplica la T-Norma(Mínimo)

def aplicar_inferencia(mu_prod: dict, mu_temp: dict, reglas: list):

#activacion = min(mu_produccion, mu_temperatura)

    activaciones = []
    for conj_prod, conj_temp, conj_dem in reglas:
        # Se asume conjuncion AND mediante T-Norma (Mínimo)
        grado_activacion = min(mu_prod.get(conj_prod, 0.0), mu_temp.get(conj_temp, 0.0))
        
        if grado_activacion > 0:
            activaciones.append((grado_activacion, conj_dem))
            
    return activaciones
