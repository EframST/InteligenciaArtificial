import numpy as np

# Definimos las funciones de activación
def identidad(y):
    return y

def escalon(y):
    return 1 if y >= 0 else 0

def sigmoide(y):
    return 1 / (1 + np.exp(-y))

def relu(y):
    return max(0, y)

# Perceptrón simple con retropropagación sin usar derivada explícita
def entrenar_perceptron(X, y, activacion, epocas=1000, tasa_aprendizaje=0.1):
    np.random.seed(42)
    pesos = np.random.rand(X.shape[1])
    umbral = np.random.rand()
    
    for epoca in range(epocas):
        for i in range(X.shape[0]):
            # Propagación hacia adelante
            suma_ponderada = np.dot(X[i], pesos) - umbral
            salida = activacion(suma_ponderada)
            
            # Cálculo del error
            error = y[i] - salida
            
            # Actualización de pesos y umbral
            pesos += tasa_aprendizaje * error * X[i]
            umbral -= tasa_aprendizaje * error
            
        # Imprimir el progreso
        if epoca % 100 == 0:
            print(f"Época {epoca}: Error medio = {np.mean(np.abs(error))}")
    
    return pesos, umbral

# Datos de ejemplo (XOR para demostración)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# Entrenamiento usando la función sigmoide
print("Entrenamiento usando la función sigmoide")
pesos, umbral = entrenar_perceptron(X, y, sigmoide)
print("Pesos finales:", pesos)
print("Umbral final:", umbral)

# Entrenamiento usando la función identidad
print("\nEntrenamiento usando la función identidad")
pesos, umbral = entrenar_perceptron(X, y, identidad)
print("Pesos finales:", pesos)
print("Umbral final:", umbral)

# Entrenamiento usando la función escalón
print("\nEntrenamiento usando la función escalón")
pesos, umbral = entrenar_perceptron(X, y, escalon)
print("Pesos finales:", pesos)
print("Umbral final:", umbral)

# Entrenamiento usando la función ReLU
print("\nEntrenamiento usando la función ReLU")
pesos, umbral = entrenar_perceptron(X, y, relu)
print("Pesos finales:", pesos)
print("Umbral final:", umbral)
