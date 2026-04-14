"""
AlgoritmoGenetico/__init__.py
Inicialización del paquete de optimización evolutiva.
"""
# Punto de entrada del paquete AG para importar desde app.py.
from .optimizador import ejecutar_ag, despacho_voraz
from .config import GENERADORES, N_GENERADORES
