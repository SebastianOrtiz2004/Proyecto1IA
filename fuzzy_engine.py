import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

def build_fuzzy_system():
    """
    Construye el Sistema de Inferencia Difusa (FIS) de tipo Mamdani.
    
    Justificación Matemática (Doctorado en IA / Inv. Operaciones): 
    Modelamos la incerteza de las variables lingüísticas a través de funciones de pertenencia 
    triangulares, operando estrictamente bajo la lógica Mamdani (operadores min/max).
    La defuzzificación se realizará deductivamente mediante el método de Centroide 
    o Centro de Gravedad para retornar el valor nítido (Crisp Value).
    """
    # 1. DEFINICIÓN DE ESPACIOS DE ESTADO (UNIVERSOS DE DISCURSO)
    # Temperatura de las máquinas: 0 a 100 °C
    temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
    # Nivel de Producción de la fábrica: 0 a 100 %
    production = ctrl.Antecedent(np.arange(0, 101, 1), 'production')
    
    # Consecuente: Demanda Energética Estimada en kW (100 a 1000)
    demand = ctrl.Consequent(np.arange(100, 1001, 1), 'demand')

    # 2. DEFINICIÓN DE LAS FUNCIONES DE PERTENENCIA (\mu(x))
    # Antecedente: Temperatura
    temperature['frio'] =     fuzz.trimf(temperature.universe, [0,   0,   45])
    temperature['normal'] =   fuzz.trimf(temperature.universe, [30,  50,  70])
    temperature['caliente'] = fuzz.trimf(temperature.universe, [55, 100, 100])

    # Antecedente: Producción
    production['bajo'] =   fuzz.trimf(production.universe, [0,   0,   45])
    production['medio'] =  fuzz.trimf(production.universe, [30,  50,  70])
    production['alto'] =   fuzz.trimf(production.universe, [55, 100, 100])

    # Consecuente: Demanda
    demand['baja'] =  fuzz.trimf(demand.universe, [100, 100, 450])
    demand['media'] = fuzz.trimf(demand.universe, [350, 550, 750])
    demand['alta'] =  fuzz.trimf(demand.universe, [650, 1000, 1000])

    # 3. BASE DE REGLAS LÓGICAS (MAMDANI)
    # Evaluación T-norma (Intersección difusa / AND)
    # Se definen reglas suficientes para asegurar una cobertura del hiperespacio.
    rule1 = ctrl.Rule(production['alto']    | temperature['caliente'], demand['alta'])
    rule2 = ctrl.Rule(production['bajo']    & temperature['frio'],     demand['baja'])
    rule3 = ctrl.Rule(production['medio']   & temperature['normal'],   demand['media'])
    rule4 = ctrl.Rule(production['bajo']    & temperature['caliente'], demand['media'])
    rule5 = ctrl.Rule(production['alto']    & temperature['normal'],   demand['alta'])
    
    # Reglas de estabilización topológica (para evitar divisiones por cero en el centroide)
    rule6 = ctrl.Rule(production['medio'] & temperature['frio'], demand['baja'])
    rule7 = ctrl.Rule(production['bajo'], demand['baja'])

    # 4. COMPILACIÓN DEL MOTOR DE INFERENCIA
    demand_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
    
    return demand_ctrl, demand


def estimate_demand(temperature_val, production_val):
    """
    Motor de ejecución del Sistema Difuso (Fase de Inferencia).
    Recibe valores 'crisp' y retorna la demanda nítida (centroide).
    """
    demand_ctrl, demand_var = build_fuzzy_system()
    demand_sim = ctrl.ControlSystemSimulation(demand_ctrl)

    demand_sim.input['temperature'] = temperature_val
    demand_sim.input['production'] = production_val

    # Cómputo del centroide: \int x * \mu(x) dx / \int \mu(x) dx
    demand_sim.compute()
    
    crisp_output = demand_sim.output['demand']
    
    return crisp_output, demand_sim, demand_var


def plot_fuzzy_result(demand_sim, demand_var):
    """
    Genera un objeto Figure de Matplotlib con el plot nativo del 
    conjunto difuso activado para ser integrado por Streamlit.
    """
    # skfuzzy dibuja automáticamente sobre plt, simplemente recuperamos la figura instanciada
    demand_var.view(sim=demand_sim)
    fig = plt.gcf()
    fig.set_size_inches(7, 3)
    plt.tight_layout()
    return fig
