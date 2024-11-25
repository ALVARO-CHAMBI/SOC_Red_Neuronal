import os
import tensorflow as tf
from preprocessing.preprocess import preprocess_data
import matplotlib.pyplot as plt
import numpy as np

# Ruta del archivo del modelo
model_path = "models/mlp_con_batch_normalization.keras"

# Cargar los datos preprocesados
data_path = "data/data.csv"
X, y = preprocess_data(data_path)

def evaluate_mlp_con_batch_norm():
    """Evalúa el modelo MLP con Batch Normalization y genera métricas."""
    if not os.path.exists(model_path):
        print(f"El modelo {model_path} no existe. Por favor, entrena el modelo primero.")
        return

    # Cargar el modelo
    print(f"Cargando el modelo desde {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Evaluar el modelo
    print("Evaluando el modelo...")
    loss, accuracy = model.evaluate(X, y, verbose=1)
    print(f"Pérdida: {loss:.4f}, Precisión: {accuracy:.4f}")

    # Predicción en ejemplos
    predictions = model.predict(X[:5])  # Predicciones para los primeros 5 ejemplos
    print("\nEjemplos de predicciones:")
    for i, prediction in enumerate(predictions):
        print(f"Entrada {i+1}: Clase Predicha = {prediction.argmax()}, Probabilidades = {prediction}")

def plot_model_performance(history_path):
    """Genera gráficas a partir de un historial guardado."""
    if not os.path.exists(history_path):
        print(f"No se encontró el historial: {history_path}")
        return

    # Cargar historial desde un archivo .npy
    history = np.load(history_path, allow_pickle=True).item()

    plt.figure(figsize=(12, 5))

    # Gráfica de la pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Pérdida Entrenamiento')
    plt.plot(history['val_loss'], label='Pérdida Validación')
    plt.title('Pérdida durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    # Gráfica de la precisión
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Precisión Entrenamiento')
    plt.plot(history['val_accuracy'], label='Precisión Validación')
    plt.title('Precisión durante el Entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    evaluate_mlp_con_batch_norm()

    # Si tienes guardado el historial, puedes graficarlo:
    # history_path = "models/mlp_con_batch_norm_history.npy"
    # plot_model_performance(history_path)
