"""
AlgoritmoGenetico/config.py
Configuración técnica de la planta y los coeficientes térmicos.
"""

# FLOTA DE GENERADORES DIÉSEL
# Formato: [Capacidad Máxima (kW), Costo Base (USD/kW·h)]
GENERADORES = [
    [300.0, 150.0], [500.0, 100.0], [200.0, 200.0], [400.0, 80.0],
    [600.0, 90.0], [150.0, 250.0], [450.0, 110.0], [350.0, 70.0],
]

N_GENERADORES = len(GENERADORES)

# COEFICIENTES DE SENSIBILIDAD TÉRMICA (USD / kW²)
COEFICIENTES_TERMICOS = [
    0.0005, 0.0003, 0.0008, 0.0002, 0.0003, 0.0010, 0.0004, 0.0001,
]
