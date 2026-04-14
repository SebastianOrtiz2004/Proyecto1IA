"""
fuzzy_engine.py — Motor de Inferencia Difusa Mamdani (IMPLEMENTACIÓN MANUAL)
=============================================================================
Implementación DESDE CERO del Sistema de Inferencia Difusa (SID) tipo Mamdani,
sin ninguna librería científica externa. Únicamente se usa:
  - math    : sqrt (defuzzificación, solo para compatibilidad futura)
  - matplotlib : para graficar el resultado difuso (solo gráficos)

NO se utiliza numpy, skfuzzy, scipy ni ninguna librería de cálculo numérico.
Todas las operaciones son Python puro (listas, bucles, comprensiones).

──────────────────────────────────────────────────────────────────────────────
FUNDAMENTO MATEMÁTICO (Defensa Universitaria):
──────────────────────────────────────────────────────────────────────────────
Funciones de pertenencia triangulares μ(x) ∈ [0,1]:
    μ_trimf(x; a,b,c) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))

Motor Mamdani (operadores mín/máx estrictos):
    T-Norma (AND)   : μ_AND(x,y)   = mín(μ_A(x), μ_B(y))
    Agregación(OR)  : μ_agg(u)     = máx sobre todas las reglas activas

Defuzzificación por Centroide (Centro de Gravedad):
    u* = Σ u · μ_agg(u)  /  Σ μ_agg(u)   (suma discreta sobre el universo)
──────────────────────────────────────────────────────────────────────────────
"""

import matplotlib
matplotlib.use('Agg')   # Evita conflictos de hilo con Streamlit
import matplotlib.pyplot as plt


# ============================================================================
# FUNCIÓN DE PERTENENCIA TRIANGULAR — IMPLEMENTACIÓN MANUAL
# ============================================================================
def _trimf(x: float, a: float, b: float, c: float) -> float:
    """
    Función de pertenencia triangular μ(x; a, b, c) ∈ [0.0, 1.0].

    Maneja correctamente los casos degenerados:
      - a == b  →  rampa derecha (escalón izquierdo): μ(a) = 1, decrece hasta c
      - b == c  →  rampa izquierda (escalón derecho): crece desde a, μ(c) = 1

    Parámetros
    ----------
    x : float  — valor a evaluar
    a : float  — punta izquierda (μ = 0 para x ≤ a, salvo a == b)
    b : float  — pico (μ = 1)
    c : float  — punta derecha  (μ = 0 para x ≥ c, salvo b == c)

    Retorna
    -------
    float ∈ [0.0, 1.0]
    """
    if x <= a:
        # Caso degenerado: rampa derecha → a==b significa que μ(a)=1
        return 1.0 if (a == b and x == a) else 0.0
    if x >= c:
        # Caso degenerado: rampa izquierda → b==c significa que μ(c)=1
        return 1.0 if (b == c and x == c) else 0.0
    if x <= b:
        # Flanco ascendente
        return 1.0 if a == b else (x - a) / (b - a)
    else:
        # Flanco descendente (x < c garantizado por rama anterior)
        return 1.0 if b == c else (c - x) / (c - b)


# ============================================================================
# CONSTRUCCIÓN DEL SISTEMA DE INFERENCIA DIFUSA
# ============================================================================
def construir_sistema_difuso():
    """
    Define y retorna la estructura completa del SID Mamdani.

    NO compila ningún grafo ni objeto externo. Devuelve un diccionario
    con los universos, las funciones de pertenencia (como tuplas a,b,c)
    y la base de reglas.

    Esta función es PURA (sin efectos secundarios) y puede cachearse
    con @st.cache_resource en app.py.

    ──────────────────────────────────────────────────────────────────────
    ESCENARIO: Planta en modo isla con 8 generadores diésel.
    Capacidad total instalada: ~2950 kW.
    ──────────────────────────────────────────────────────────────────────

    Retorna
    -------
    sistema : dict   — SID completo con universos, MFs y reglas
    None             — segundo valor mantenido por compatibilidad con app.py
    """

    # ── 1. UNIVERSOS DE DISCURSO ────────────────────────────────────────
    # Temperatura exterior: 0..100 °C  (paso 1)
    universo_temp = list(range(0, 101))
    # Carga productiva:     0..100 %   (paso 1)
    universo_prod = list(range(0, 101))
    # Demanda estimada:  200..2200 kW  (paso 1)
    universo_dem  = list(range(200, 2201))

    # ── 2. FUNCIONES DE PERTENENCIA TRIANGULARES ────────────────────────
    # Antecedente 1: Temperatura (a, b, c)
    #   frio(0,0,45)   normal(30,50,70)   caliente(55,100,100)
    mfs_temp = {
        'frio':     (0,   0,   45),
        'normal':   (30,  50,  70),
        'caliente': (55, 100, 100),
    }

    # Antecedente 2: Producción (a, b, c)
    #   bajo(0,0,45)   medio(30,50,70)   alto(55,100,100)
    mfs_prod = {
        'bajo':  (0,   0,   45),
        'medio': (30,  50,  70),
        'alto':  (55, 100, 100),
    }

    # Consecuente: Demanda kW (a, b, c)
    #   baja(200,200,800)   media(600,1100,1600)   alta(1300,2200,2200)
    mfs_dem = {
        'baja':  (200,  200,   800),
        'media': (600,  1100, 1600),
        'alta':  (1300, 2200, 2200),
    }

    # ── 3. BASE DE REGLAS MAMDANI (9 reglas — cobertura completa) ───────
    #
    # Tabla de cobertura (3 conjuntos × 3 conjuntos = 9 combinaciones):
    #
    #   Producción \ Temp. │  Frío           │  Normal         │  Caliente       │
    #   ───────────────────┼─────────────────┼─────────────────┼─────────────────┤
    #   Alto               │ regla1 (alta)   │ regla2 (alta)   │ regla3 (alta)   │
    #   Medio              │ regla4 (baja)   │ regla5 (media)  │ regla6 (alta)   │
    #   Bajo               │ regla7 (baja)   │ regla8 (baja)   │ regla9 (media)  │
    #
    # Formato de cada regla: (conjunto_produccion, conjunto_temperatura, conjunto_demanda)
    # T-norma AND = mín(μ_prod, μ_temp)  →  lógica Mamdani estricta
    # ─────────────────────────────────────────────────────────────────────
    reglas = [
        # Bloque 1: Producción ALTA → demanda siempre ALTA
        ('alto',  'frio',     'alta'),
        ('alto',  'normal',   'alta'),
        ('alto',  'caliente', 'alta'),
        # Bloque 2: Producción MEDIA → demanda varía con temperatura
        ('medio', 'frio',     'baja'),
        ('medio', 'normal',   'media'),
        ('medio', 'caliente', 'alta'),
        # Bloque 3: Producción BAJA → demanda generalmente baja
        ('bajo',  'frio',     'baja'),
        ('bajo',  'normal',   'baja'),
        ('bajo',  'caliente', 'media'),
    ]

    sistema = {
        'universo_temp': universo_temp,
        'universo_prod': universo_prod,
        'universo_dem':  universo_dem,
        'mfs_temp':      mfs_temp,
        'mfs_prod':      mfs_prod,
        'mfs_dem':       mfs_dem,
        'reglas':        reglas,
    }

    # Retorna (sistema, None) para mantener la interfaz:
    # sistema_control, var_demanda = construir_sistema_difuso()
    return sistema, None


# ============================================================================
# INFERENCIA DIFUSA — FASE DE CÓMPUTO
# ============================================================================
def estimar_demanda(sistema: dict, _var_demanda, valor_temperatura: float, valor_produccion: float):
    """
    Aplica inferencia Mamdani sobre el sistema difuso para datos crisp.

    Proceso completo (todo Python puro):
      1. Fuzzificación: calcular μ para cada antecedente
      2. Inferencia:    T-norma mín(AND) por cada regla
      3. Agregación:    máx sobre todas las reglas activas (por punto del universo)
      4. Defuzzificación: centroide discreto

    Parámetros
    ----------
    sistema          : dict   — SID retornado por construir_sistema_difuso()
    _var_demanda     : any    — ignorado (compatibilidad con app.py)
    valor_temperatura : float — temperatura en °C [0, 100]
    valor_produccion  : float — carga productiva en % [0, 100]

    Retorna
    -------
    salida_crisp : float — demanda defuzzificada en kW (centroide)
    simulacion   : dict  — estado de la inferencia para graficar
    """
    # Recorte defensivo de entradas al universo de discurso
    temperatura = min(100.0, max(0.0, float(valor_temperatura)))
    produccion  = min(100.0, max(0.0, float(valor_produccion)))

    mfs_temp = sistema['mfs_temp']
    mfs_prod = sistema['mfs_prod']
    mfs_dem  = sistema['mfs_dem']
    reglas   = sistema['reglas']
    universo_dem = sistema['universo_dem']

    # ── FUZZIFICACIÓN ─────────────────────────────────────────────────
    # Evaluar μ de cada conjunto lingüístico de los antecedentes
    mu_temp = {nombre: _trimf(temperatura, *params) for nombre, params in mfs_temp.items()}
    mu_prod = {nombre: _trimf(produccion,  *params) for nombre, params in mfs_prod.items()}

    # ── INFERENCIA + AGREGACIÓN (Máx-Mín Mamdani) ────────────────────
    # Para cada punto u del universo de demanda:
    #   μ_agg(u) = máx sobre todas las reglas de:
    #              mín(activación_regla, μ_dem_set(u))
    N = len(universo_dem)
    mu_agregada = [0.0] * N

    for conj_prod, conj_temp, conj_dem in reglas:
        # Activación de la regla: T-norma mín (AND estricto Mamdani)
        activacion = min(mu_prod[conj_prod], mu_temp[conj_temp])

        # Optimización: saltar reglas con activación nula
        if activacion <= 0.0:
            continue

        a, b, c = mfs_dem[conj_dem]
        for idx in range(N):
            u = universo_dem[idx]
            # Clipear la MF del consecuente al nivel de activación
            mu_clipped = min(activacion, _trimf(u, a, b, c))
            # Agregar (máx) sobre todas las reglas
            if mu_clipped > mu_agregada[idx]:
                mu_agregada[idx] = mu_clipped

    # ── DEFUZZIFICACIÓN POR CENTROIDE ─────────────────────────────────
    # u* = Σ(u · μ_agg(u)) / Σ(μ_agg(u))   — suma discreta sobre el universo
    numerador   = sum(universo_dem[i] * mu_agregada[i] for i in range(N))
    denominador = sum(mu_agregada)

    if denominador == 0.0:
        # Seguridad: si el área agregada es cero, usar el centro del universo
        salida_crisp = (universo_dem[0] + universo_dem[-1]) / 2.0
    else:
        salida_crisp = numerador / denominador

    # Empaquetar estado de la simulación para graficar
    simulacion = {
        'mu_agregada':  mu_agregada,
        'universo_dem': universo_dem,
        'mfs_dem':      mfs_dem,
        'salida_crisp': salida_crisp,
        'mu_temp':      mu_temp,
        'mu_prod':      mu_prod,
        'temperatura':  temperatura,
        'produccion':   produccion,
    }

    return salida_crisp, simulacion


# ============================================================================
# GRÁFICO DEL CONJUNTO DIFUSO DEFUZZIFICADO — MATPLOTLIB PURO
# ============================================================================
def graficar_resultado_difuso(simulacion: dict, _var_demanda) -> plt.Figure:
    """
    Genera una Figure de Matplotlib con:
      - Las 3 funciones de pertenencia del consecuente (baja/media/alta)
      - El área agregada (resultado de la inferencia Mamdani)
      - La línea vertical del centroide defuzzificado (u*)

    Toda la construcción del gráfico es matplotlib puro, listas nativas;
    no se usa skfuzzy ni numpy en ningún paso.

    Parámetros
    ----------
    simulacion   : dict — estado retornado por estimar_demanda()
    _var_demanda : any  — ignorado (compatibilidad API con app.py / charts.py)

    Retorna
    -------
    figura : plt.Figure
    """
    universo_dem = simulacion['universo_dem']
    mfs_dem      = simulacion['mfs_dem']
    mu_agregada  = simulacion['mu_agregada']
    salida_crisp = simulacion['salida_crisp']

    figura, eje = plt.subplots(figsize=(7, 3))

    # Paleta de colores por conjunto lingüístico
    colores = {'baja': '#6699ff', 'media': '#f2c94c', 'alta': '#ff6b6b'}

    # Dibujar las MF originales del consecuente (sin clipear)
    for nombre, (a, b, c) in mfs_dem.items():
        mu_vals = [_trimf(u, a, b, c) for u in universo_dem]
        eje.plot(universo_dem, mu_vals,
                 label=nombre.capitalize(),
                 color=colores[nombre],
                 linewidth=1.5, alpha=0.65)

    # Dibujar el área agregada de la inferencia (región defuzzificada)
    eje.fill_between(universo_dem, mu_agregada,
                     alpha=0.30, color='#00ffcc',
                     label='Área activada (Mamdani)')

    # Línea vertical del centroide u* (resultado defuzzificado)
    eje.axvline(salida_crisp,
                color='white', linestyle='--', linewidth=1.5,
                label=f'Centroide u* = {salida_crisp:.1f} kW')

    eje.set_xlabel('Demanda (kW)', color='#ddd')
    eje.set_ylabel('Pertenencia  μ(u)', color='#ddd')
    eje.set_title('Consecuente Mamdani — Defuzzificado por Centroide', color='#ddd')
    eje.set_xlim(universo_dem[0], universo_dem[-1])
    eje.set_ylim(0, 1.12)
    eje.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    return figura
