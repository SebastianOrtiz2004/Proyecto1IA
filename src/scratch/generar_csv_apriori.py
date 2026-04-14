import os
import csv
import random
from datetime import datetime, timedelta

# Reutilizar algo de lógica para que la demanda tenga sentido físico
def _trimf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x <= b: return (x - a) / (b - a)
    return (c - x) / (c - b)

def generar_dataset_historico_apriori(n_muestras=100):
    os.makedirs("DataSet", exist_ok=True)
    ruta_archivo = os.path.join("DataSet", "historico_planta.csv")
    
    inicio_fecha = datetime(2026, 4, 1, 8, 0, 0)
    # Incluimos Demanda_Real_kW para que Apriori tenga algo que aprender
    headers = ["ID", "Fecha_Hora", "Temperatura_Ambiente_C", "Carga_Industrial_Pct", "Demanda_Real_kW"]
    
    print(f"Generando {n_muestras} registros con Demanda en {ruta_archivo}...")
    
    with open(ruta_archivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(1, n_muestras + 1):
            fecha_actual = inicio_fecha + timedelta(hours=i-1)
            hora = fecha_actual.hour
            
            # Temp y Carga con lógica horaria
            temp_base = 22.0 + 8.0 * (1.0 - abs(hora - 14) / 12.0) 
            temp = round(temp_base + random.uniform(-2, 2), 2)
            
            carga_base = 75.0 if 8 <= hora <= 18 else 45.0
            carga = round(carga_base + random.uniform(-15, 15), 2)
            carga = max(5.0, min(98.0, carga))
            
            # Generar una Demanda "Real" basada en una lógica oculta que Apriori debe descubrir
            # Si Carga Alta -> Demanda Alta (~1800 kW)
            # Si Temp Alta -> Demanda sube un poco (+200 kW)
            # Si Carga Baja -> Demanda Baja (~500 kW)
            demanda_base = 400 + (carga / 100 * 1600)
            efecto_temp = (temp / 40 * 250) if temp > 25 else 0
            demanda = round(demanda_base + efecto_temp + random.uniform(-50, 50), 2)
            demanda = max(200, min(2200, demanda))
            
            writer.writerow([i, fecha_actual.strftime("%Y-%m-%d %H:%M:%S"), temp, carga, demanda])

    print("¡Dataset para Apriori generado exitosamente!")

if __name__ == "__main__":
    generar_dataset_historico_apriori(100)
