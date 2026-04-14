
# Salida final del SID: valor unico de demanda en kW.
def centroide(mu_agregada: list, universo_dem: list) -> float:
    
    #u* = Σ(u · μ_agg(u)) / Σ(μ_agg(u))
    
    numerador = sum(universo_dem[i] * mu_agregada[i] for i in range(len(universo_dem)))
    denominador = sum(mu_agregada)
    
    if denominador == 0.0:
        return (universo_dem[0] + universo_dem[-1]) / 2.0
    return numerador / denominador
