import os
import csv
import random
from datetime import datetime, timedelta

# Variante con mayor dispersion para estresar el minado de reglas.
def generar_dataset_historico_apriori_v2(n_muestras=100):
    os.makedirs("DataSet", exist_ok=True)
    ruta_archivo = os.path.join("DataSet", "historico_planta.csv")
    
    inicio_fecha = datetime(2026, 4, 1, 8, 0, 0)
    headers = ["ID", "Fecha_Hora", "Temperatura_Ambiente_C", "Carga_Industrial_Pct", "Demanda_Real_kW"]
    
    print(f"Generando {n_muestras} registros estocásticamente diversos en {ruta_archivo}...")
    
    with open(ruta_archivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Para garantizar cobertura de reglas, forzaremos 11 combinaciones base 
        # y el resto será aleatorio puro (dispersión uniforme)
        for i in range(1, n_muestras + 1):
            fecha_actual = inicio_fecha + timedelta(hours=i-1)
            
            # Aumentamos la varianza de Temperatura (10°C a 45°C)
            temp = round(random.uniform(12.0, 42.0), 2)
            
            # Aumentamos la varianza de Carga (5% a 95%)
            carga = round(random.uniform(5.0, 95.0), 2)
            
            # Lógica de Demanda Proporcional (Target para Apriori)
            # Definimos un comportamiento determinista con ruido Gaussiano
            if carga < 35:
                demanda_base = 450
            elif carga < 65:
                demanda_base = 1100
            else:
                demanda_base = 1850
            
            # Influencia Térmica Cuadrática para mayor realismo físico
            efecto_temp = 2.5 * (max(0, temp - 25)**2)
            
            demanda = round(demanda_base + efecto_temp + random.gauss(0, 40), 1)
            demanda = max(200, min(2300, demanda))
            
            writer.writerow([i, fecha_actual.strftime("%Y-%m-%d %H:%M:%S"), temp, carga, demanda])

    print("¡Dataset diversificado generado exitosamente!")

if __name__ == "__main__":
    generar_dataset_historico_apriori_v2(100)
