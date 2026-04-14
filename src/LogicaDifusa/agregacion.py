from .fuzzificacion import trimf

# Junta todas las reglas activas en una sola curva de salida.
def agregar_reglas(activaciones: list, mfs_dem: dict, universo_dem: list):

#Agregación de Mamdani. μ_agg(u) = máx sobre todas las reglas de: mín(activación, mu_demanda(u))
    n = len(universo_dem)
    mu_agregada = [0.0] * n
    
    for activacion, conj_dem in activaciones:
        a, b, c = mfs_dem[conj_dem]
        
        for idx in range(n):
            u = universo_dem[idx]
            # Clipear la MF del consecuente al nivel de activación
            mu_clipped = min(activacion, trimf(u, a, b, c))
            # S-Norma (Máximo) para la agregación
            if mu_clipped > mu_agregada[idx]:
                mu_agregada[idx] = mu_clipped
                
    return mu_agregada
