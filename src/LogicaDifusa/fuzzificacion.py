# Aqui se transforma entrada crisp -> grados difusos.

def trimf(x: float, a: float, b: float, c: float) -> float:

# Función de pertenencia triangular mu(x; a, b, c) en [0, 1].

    if x <= a:
        return 1.0 if (a == b and x == a) else 0.0
    if x >= c:
        return 1.0 if (b == c and x == c) else 0.0
    if x <= b:
        return 1.0 if a == b else (x - a) / (b - a)
    else:
        return 1.0 if b == c else (c - x) / (c - b)

def calcular_pertenencia(valor: float, mfs: dict) -> dict:
    return {nombre: trimf(valor, *params) for nombre, params in mfs.items()}
