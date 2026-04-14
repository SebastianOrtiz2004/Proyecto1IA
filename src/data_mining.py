"""
data_mining.py — Algoritmo Apriori para Minería de Reglas de Asociación (PYTHON PURO)
===================================================================================
Implementación desde cero del algoritmo Apriori para extraer reglas de decisión
basadas en el dataset histórico de la planta. 

Proceso:
  1. Carga del CSV.
  2. Discretización: Convertir valores numéricos a etiquetas (Frio, Normal, Caliente, etc.)
  3. Generación de Itemsets Frecuentes (Soporte).
  4. Generación de Reglas de Asociación (Confianza).
  5. Filtrado de reglas útiles para el motor difuso.
"""

import csv
import os

# Ruta al dataset desde la carpeta src/
RUTA_DATASET = "src/DataSet/historico_planta.csv"

def cargar_datos_csv(ruta=None):
    """Lee el CSV y devuelve una lista de diccionarios."""
    if ruta is None:
        ruta = RUTA_DATASET
    datos = []
    if not os.path.exists(ruta):
        print(f"Error: No se encontró el dataset en {os.path.abspath(ruta)}")
        return datos
    with open(ruta, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for fila in reader:
            datos.append({
                'temp': float(fila['Temperatura_Ambiente_C']),
                'carga': float(fila['Carga_Industrial_Pct']),
                'demanda': float(fila['Demanda_Real_kW'])
            })
    return datos

def discretizar_datos(datos):
    """Convierte valores continuos en categorías para Apriori."""
    transacciones = []
    for d in datos:
        t = []
        # Temperatura
        if d['temp'] < 20: t.append('temp_frio')
        elif d['temp'] < 30: t.append('temp_normal')
        else: t.append('temp_caliente')
        
        # Carga
        if d['carga'] < 40: t.append('prod_bajo')
        elif d['carga'] < 70: t.append('prod_medio')
        else: t.append('prod_alto')
        
        # Demanda
        if d['demanda'] < 800: t.append('dem_baja')
        elif d['demanda'] < 1500: t.append('dem_media')
        else: t.append('dem_alta')
        
        transacciones.append(t)
    return transacciones

def obtener_itemsets_frecuentes(transacciones, soporte_min=0.01): # Bajado a 1%
    """Versión mejorada de Apriori para itemsets de tamaño 1, 2 y 3."""
    total = len(transacciones)
    contadores = {}
    
    # 1-itemsets
    for t in transacciones:
        for item in t:
            contadores[frozenset([item])] = contadores.get(frozenset([item]), 0) + 1
            
    # Itemsets de tamaño 2 y 3 (generación exhaustiva para N pequeo)
    for t in transacciones:
        if len(t) < 2: continue
        # Combinaciones de 2
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                itemset2 = frozenset([t[i], t[j]])
                contadores[itemset2] = contadores.get(itemset2, 0) + 1
                
                # Combinaciones de 3
                for k in range(j + 1, len(t)):
                    itemset3 = frozenset([t[i], t[j], t[k]])
                    contadores[itemset3] = contadores.get(itemset3, 0) + 1

    # Filtrar por soporte
    frecuentes = {itemset: count/total for itemset, count in contadores.items() if count/total >= soporte_min}
    return frecuentes

def extraer_reglas_fuzzy(frecuentes, confianza_min=0.35): # Bajado a 35% para mayor cobertura
    """
    Extrae reglas del tipo: {temp_X, prod_Y} => {dem_Z}
    Filtra solo las reglas que sirven para el motor difuso.
    """
    reglas = []
    
    # Buscamos itemsets de tamao 3 que tengan (1 temp, 1 prod, 1 dem)
    for itemset, soporte in frecuentes.items():
        if len(itemset) == 3:
            items = list(itemset)
            temp = next((i for i in items if i.startswith('temp_')), None)
            prod = next((i for i in items if i.startswith('prod_')), None)
            dem = next((i for i in items if i.startswith('dem_')), None)
            
            if temp and prod and dem:
                # Antecedente: {temp, prod}
                antecedente = frozenset([temp, prod])
                sop_antecedente = frecuentes.get(antecedente, 0)
                
                if sop_antecedente > 0:
                    confianza = soporte / sop_antecedente
                    if confianza >= confianza_min:
                        reglas.append({
                            'temp': temp.replace('temp_', ''),
                            'prod': prod.replace('prod_', ''),
                            'dem': dem.replace('dem_', ''),
                            'soporte': soporte,
                            'confianza': confianza
                        })
    return reglas

def minar_reglas_proyecto():
    """Función principal para ser llamada desde app.py o fuzzy_engine.py"""
    datos = cargar_datos_csv()
    if not datos:
        return []
    
    transacciones = discretizar_datos(datos)
    # Hiperparámetros de minería optimizados para rigor académico
    frecuentes = obtener_itemsets_frecuentes(transacciones, soporte_min=0.02)
    reglas = extraer_reglas_fuzzy(frecuentes, confianza_min=0.35)
    
    # Eliminar duplicados lógicos (si existen múltiples consecuentes para el mismo antecedente,
    # nos quedamos con el de mayor confianza)
    unicas = {}
    for r in reglas:
        clave = (r['temp'], r['prod'])
        if clave not in unicas or r['confianza'] > unicas[clave]['confianza']:
            unicas[clave] = r
            
    resultado = list(unicas.values())
    resultado.sort(key=lambda x: x['confianza'], reverse=True)
    return resultado

if __name__ == "__main__":
    reglas = minar_reglas_proyecto()
    print(f"Se encontraron {len(reglas)} reglas de asociación fuertes:")
    for r in reglas:
        print(f"SI [Temp={r['temp']}] Y [Prod={r['prod']}] ENTONCES [Demanda={r['dem']}] (Conf: {r['confianza']:.2f})")
