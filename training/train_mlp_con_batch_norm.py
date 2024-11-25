import os
from tensorflow.keras.optimizers import Adam
from preprocessing.preprocess import preprocess_data
from model_definitions.mlp_con_batch_normalization import create_mlp_with_batch_norm

# Cargar datos
X, y = preprocess_data("data/data.csv")

# Crear el modelo
model = create_mlp_with_batch_norm(input_dim=X.shape[1], output_dim=y.shape[1])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
print("Entrenando modelo MLP con Batch Normalization...")
history = model.fit(X, y, epochs=50, batch_size=4, validation_split=0.2)

# Guardar el modelo
os.makedirs("models", exist_ok=True)
model.save("models/mlp_con_batch_normalization.keras")
print("Modelo MLP con Batch Normalization guardado en 'models/mlp_con_batch_normalization.keras'")
