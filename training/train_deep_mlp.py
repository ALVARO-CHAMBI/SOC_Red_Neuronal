import os
from tensorflow.keras.optimizers import Adam
from preprocessing.preprocess import preprocess_data
from models_definitions.deep_mlp import create_deep_mlp
import numpy as np

# Ruta de datos y modelos
data_path = "data/data.csv"
model_path = "models/deep_mlp.keras"
history_path = "models/deep_mlp_history.npy"

# Verificar que el archivo de datos exista
if not os.path.exists(data_path):
    raise FileNotFoundError(f"El archivo de datos no existe en la ruta: {data_path}")

# Cargar datos
print("Cargando y preprocesando los datos...")
X, y = preprocess_data(data_path)

# Crear el modelo
print("Creando el modelo Deep MLP...")
model = create_deep_mlp(input_dim=X.shape[1], output_dim=y.shape[1])

# Compilar el modelo
print("Compilando el modelo...")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
print("Entrenando modelo Deep MLP...")
history = model.fit(X, y, epochs=50, batch_size=4, validation_split=0.2)

# Crear carpeta para guardar el modelo y el historial
os.makedirs("models", exist_ok=True)

# Guardar el modelo
print(f"Guardando el modelo en: {model_path}...")
model.save(model_path)

# Guardar el historial del entrenamiento
print(f"Guardando el historial del entrenamiento en: {history_path}...")
np.save(history_path, history.history)

print("Entrenamiento completo y modelos guardados correctamente.")
