import os
import csv
import random
from datetime import datetime, timedelta

# Script rapido para generar historico base de pruebas.
def generar_dataset_historico(n_muestras=100):
    # Definir ruta (ya creada previamente con mkdir)
    ruta_archivo = os.path.join("DataSet", "historico_planta.csv")
    
    # Parámetros de simulación realistas
    inicio_fecha = datetime(2026, 4, 1, 8, 0, 0) # Empieza 1 de Abril a las 8 AM
    headers = ["ID", "Fecha_Hora", "Temperatura_Ambiente_C", "Carga_Industrial_Pct"]
    
    print(f"Generando {n_muestras} registros en {ruta_archivo}...")
    
    with open(ruta_archivo, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for i in range(1, n_muestras + 1):
            # Simular avance del tiempo (1 hora por registro)
            fecha_actual = inicio_fecha + timedelta(hours=i-1)
            str_fecha = fecha_actual.strftime("%Y-%m-%d %H:%M:%S")
            
            # Dinámica de Temperatura (más calor al mediodía, menos noche)
            hora = fecha_actual.hour
            # Una onda simple para simular ciclo diario: base 20°C + oscilación
            temp_base = 22.0 + 8.0 * (1.0 - abs(hora - 14) / 12.0) 
            temperatura = round(temp_base + random.uniform(-2, 2), 2)
            
            # Dinámica de Carga (más carga en horario laboral 8-18)
            if 8 <= hora <= 18:
                carga_base = 75.0
            else:
                carga_base = 45.0
            carga = round(carga_base + random.uniform(-15, 15), 2)
            carga = max(5.0, min(98.0, carga)) # Mantener en rango %
            
            writer.writerow([i, str_fecha, temperatura, carga])

    print("Dataset generado")

if __name__ == "__main__":
    generar_dataset_historico(100)
