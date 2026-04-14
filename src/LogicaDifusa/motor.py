"""
LogicaDifusa/motor.py
Orquestador del Sistema de Inferencia Difusa Modular.
Une todas las fases (Fuzzificación, Inferencia, Agregación, Defuzzificación).
"""

import matplotlib.pyplot as plt
from data_mining import minar_reglas_proyecto
from .fuzzificacion import calcular_pertenencia, trimf
from .inferencia import aplicar_inferencia
from .agregacion import agregar_reglas
from .defuzzificacion import centroide

def construir_sistema_difuso():
    """Define los universos y MFs (Sincronizado con Apriori)."""
    universo_temp = list(range(0, 101))
    universo_prod = list(range(0, 101))
    universo_dem  = list(range(200, 2201))

    mfs_temp = {
        'frio':     (0,   10,  22),
        'normal':   (18,  25,  32),
        'caliente': (28,  40, 100),
    }

    mfs_prod = {
        'bajo':  (0,   20,  45),
        'medio': (35,  55,  75),
        'alto':  (65,  85, 100),
    }

    mfs_dem = {
        'baja':  (200,  200,   800),
        'media': (600,  1100, 1600),
        'alta':  (1300, 2200, 2200),
    }

    reglas_minadas = minar_reglas_proyecto()
    reglas = [(r['prod'], r['temp'], r['dem']) for r in reglas_minadas]

    if not reglas:
        reglas = [('medio', 'normal', 'media'), ('alto', 'normal', 'alta'), ('bajo', 'normal', 'baja')]

    sistema = {
        'universo_temp': universo_temp, 'universo_prod': universo_prod, 'universo_dem': universo_dem,
        'mfs_temp': mfs_temp, 'mfs_prod': mfs_prod, 'mfs_dem': mfs_dem,
        'reglas': reglas, 'reglas_detalles': reglas_minadas
    }
    return sistema, None

def estimar_demanda(sistema, _var_demanda, temp: float, prod: float):
    """Ejecuta las 4 fases de la lógica difusa paso a paso."""
    # 1. FUZZIFICACIÓN
    mu_temp = calcular_pertenencia(temp, sistema['mfs_temp'])
    mu_prod = calcular_pertenencia(prod, sistema['mfs_prod'])

    # 2. INFERENCIA (MAMDANI)
    activaciones = aplicar_inferencia(mu_prod, mu_temp, sistema['reglas'])

    # 3. AGREGACIÓN (MAX-MIN)
    mu_agregada = agregar_reglas(activaciones, sistema['mfs_dem'], sistema['universo_dem'])

    # 4. DEFUZZIFICACIÓN (CENTROIDE)
    salida_crisp = centroide(mu_agregada, sistema['universo_dem'])

    simulacion = {
        'mu_agregada': mu_agregada, 'universo_dem': sistema['universo_dem'],
        'mfs_dem': sistema['mfs_dem'], 'salida_crisp': salida_crisp,
        'mu_temp': mu_temp, 'mu_prod': mu_prod, 'temperatura': temp, 'produccion': prod
    }
    return salida_crisp, simulacion

def graficar_resultado_difuso(simulacion: dict, _v) -> plt.Figure:
    """Mantiene compatibilidad con la UI de Streamlit."""
    universo_dem = simulacion['universo_dem']
    mfs_dem      = simulacion['mfs_dem']
    mu_agregada  = simulacion['mu_agregada']
    salida_crisp = simulacion['salida_crisp']

    figura, eje = plt.subplots(figsize=(7, 3))
    colores = {'baja': '#6699ff', 'media': '#f2c94c', 'alta': '#ff6b6b'}

    for nombre, (a, b, c) in mfs_dem.items():
        mu_vals = [trimf(u, a, b, c) for u in universo_dem]
        eje.plot(universo_dem, mu_vals, label=nombre.capitalize(), color=colores[nombre], linewidth=1.5, alpha=0.6)

    eje.fill_between(universo_dem, mu_agregada, alpha=0.30, color='#00ffcc', label='Área agregada')
    eje.axvline(salida_crisp, color='white', linestyle='--', label=f'u* = {salida_crisp:.1f} kW')

    eje.set_title('Sistema de Inferencia Difusa (Fases Modularizadas)', color='#ddd')
    eje.legend(fontsize=8)
    plt.tight_layout()
    return figura
