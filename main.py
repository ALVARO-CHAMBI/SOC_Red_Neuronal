# import re
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import to_categorical
#
# # Cargar el CSV
# csv_path = "data_verify.csv"  # Ruta del CSV
# df = pd.read_csv(csv_path)
#
#
# # Preprocesar 'Lista de cortes' en vectores numéricos
# def process_cut_list(cut_list):
#     if pd.isna(cut_list) or not cut_list.strip():
#         return np.zeros(10)  # Supongamos 10 tipos posibles de cortes
#     pattern = r"\((\d+)\*(\d+\.?\d*)\)\*(\d+)"
#     matches = re.findall(pattern, cut_list)
#     vector = np.zeros(10)
#     for i, match in enumerate(matches):
#         cantidad = int(match[2])
#         vector[i] = cantidad
#     return vector
#
#
# # Aplicar preprocesamiento a la columna 'Lista de cortes'
# X_cuts = np.array([process_cut_list(cuts) for cuts in df['Lista de cortes']])
#
# # Escalar características numéricas: Área Usada % y Área Sobrante %
# scaler = MinMaxScaler()
# X_area = scaler.fit_transform(df[['Area Usada %', 'Area Sobrante %']])
#
# # Combinar las características en un solo array
# X = np.hstack((X_cuts, X_area))
#
# # Preparar la salida (y)
# y = to_categorical(df['ID'] - 1)  # IDs como clases desde 0
#
# # Dividir los datos en entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Información de verificación
# print(f"Shape de X_train: {X_train.shape}")
# print(f"Shape de y_train: {y_train.shape}")
# print(f"Shape de X_test: {X_test.shape}")
# print(f"Shape de y_test: {y_test.shape}")
#
# # -----------------------------------------------
# # Modelo 3: MLP con Dropout
# # -----------------------------------------------
# # model3 = Sequential([
# #     Dense(128, input_dim=X_train.shape[1], activation='relu'),
# #     Dropout(0.2),  # Regularización para evitar sobreajuste
# #     Dense(64, activation='relu'),
# #     Dense(y_train.shape[1], activation='softmax')  # Salida para clasificación multiclase
# # ])
# #
# # model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # print("Entrenando Modelo 3 (MLP con Dropout)...")
# # model3.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))
# # loss3, acc3 = model3.evaluate(X_test, y_test)
# # print(f"Modelo 3 - Precisión: {acc3}, Pérdida: {loss3}")
#
#
#
#
#
#
#
#
#
# model3 = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu'),
#     BatchNormalization(),  # Normalización por lotes
#     Dropout(0.2),  # Regularización
#     Dense(64, activation='relu'),
#     Dense(y_train.shape[1], activation='softmax')  # Capa de salida
# ])
#
# # Compilar el modelo
# optimizer = Adam(learning_rate=0.001)
# model3.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Callbacks para mejorar el entrenamiento
# early_stopping = EarlyStopping(
#     monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
# )
# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
# )
#
# # Entrenar el modelo con validación
# print("Entrenando Modelo 3 (Mejorado)...")
# history = model3.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=4,
#     validation_data=(X_test, y_test),
#     callbacks=[early_stopping, reduce_lr]
# )
#
# # Evaluar el modelo
# loss3, acc3 = model3.evaluate(X_test, y_test)
# print(f"Modelo 3 - Precisión: {acc3:.4f}, Pérdida: {loss3:.4f}")
#
# # -------------------------------------
# # Graficar el proceso de entrenamiento
# # -------------------------------------
#
# # 1. Gráfica de pérdida (loss)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
# plt.plot(history.history['val_loss'], label='Pérdida Validación')
# plt.title('Pérdida durante el Entrenamiento')
# plt.xlabel('Épocas')
# plt.ylabel('Pérdida')
# plt.legend()
#
# # 2. Gráfica de precisión (accuracy)
# plt.subplot(1, 2, 2)
# plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
# plt.plot(history.history['val_accuracy'], label='Precisión Validación')
# plt.title('Precisión durante el Entrenamiento')
# plt.xlabel('Épocas')
# plt.ylabel('Precisión')
# plt.legend()
#
# plt.show()
#
#
# # # -----------------------------------------------
# # # Modelo 4: MLP con Batch Normalization
# # # -----------------------------------------------
# # model4 = Sequential([
# #     Dense(128, input_dim=X_train.shape[1], activation='relu'),
# #     BatchNormalization(),  # Normalización para estabilizar el entrenamiento
# #     Dense(64, activation='relu'),
# #     Dense(y_train.shape[1], activation='softmax')
# # ])
# #
# # model4.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # print("Entrenando Modelo 4 (MLP con Batch Normalization)...")
# # model4.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))
# # loss4, acc4 = model4.evaluate(X_test, y_test)
# # print(f"Modelo 4 - Precisión: {acc4}, Pérdida: {loss4}")
# #
# # # -----------------------------------------------
# # # Modelo 5: Deep MLP con 3 capas ocultas
# # # -----------------------------------------------
# # model5 = Sequential([
# #     Dense(128, input_dim=X_train.shape[1], activation='relu'),
# #     Dense(128, activation='relu'),
# #     Dense(64, activation='relu'),
# #     Dense(y_train.shape[1], activation='softmax')
# # ])
# #
# # model5.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # print("Entrenando Modelo 5 (Deep MLP)...")
# # model5.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))
# # loss5, acc5 = model5.evaluate(X_test, y_test)
# # print(f"Modelo 5 - Precisión: {acc5}, Pérdida: {loss5}")
#
# # -----------------------------------------------
# # Comparación de Modelos
# # -----------------------------------------------
# print("\nResultados de los Modelos:")
# print(f"Modelo 3 (Dropout): Precisión = {acc3:.2f}, Pérdida = {loss3:.2f}")
# # print(f"Modelo 4 (Batch Normalization): Precisión = {acc4:.2f}, Pérdida = {loss4:.2f}")
# # print(f"Modelo 5 (Deep MLP): Precisión = {acc5:.2f}, Pérdida = {loss5:.2f}")
# #
# # # -----------------------------------------------
# # Guardar el la informacion del modelo para utilizarlo
# # -----------------------------------------------
#
# model3.save("mlp_dropout_model.keras")


import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Cargar el CSV
csv_path = "data_verify.csv"  # Ruta del CSV
df = pd.read_csv(csv_path)

# Preprocesar 'Lista de cortes' en vectores numéricos
def process_cut_list(cut_list):
    if pd.isna(cut_list) or not cut_list.strip():
        return np.zeros(10)  # Supongamos 10 tipos posibles de cortes
    pattern = r"\((\d+)\*(\d+\.?\d*)\)\*(\d+)"
    matches = re.findall(pattern, cut_list)
    vector = np.zeros(10)
    for i, match in enumerate(matches):
        cantidad = int(match[2])
        vector[i] = cantidad
    return vector

# Aplicar preprocesamiento a la columna 'Lista de cortes'
X_cuts = np.array([process_cut_list(cuts) for cuts in df['Lista de cortes']])

# Escalar características numéricas: Área Usada % y Área Sobrante %
scaler = MinMaxScaler()
X_area = scaler.fit_transform(df[['Area Usada %', 'Area Sobrante %']])

# Combinar las características en un solo array
X = np.hstack((X_cuts, X_area))

# Preparar la salida (y)
y = to_categorical(df['ID'] - 1)  # IDs como clases desde 0

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Información de verificación
print(f"Shape de X_train: {X_train.shape}")
print(f"Shape de y_train: {y_train.shape}")
print(f"Shape de X_test: {X_test.shape}")
print(f"Shape de y_test: {y_test.shape}")

# -----------------------------------------------
# Modelo: MLP con Dropout y BatchNormalization
# -----------------------------------------------
model3 = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='softmax')  # Capa de salida
])

# Compilar el modelo con una tasa de aprendizaje más lenta
optimizer = Adam(learning_rate=0.0005)
model3.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks mejorados
early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1
)

# Entrenamiento del modelo
print("Entrenando Modelo 3 (Mejorado)...")
history = model3.fit(
    X_train, y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluar el modelo
loss3, acc3 = model3.evaluate(X_test, y_test)
print(f"Modelo 3 - Precisión: {acc3:.4f}, Pérdida: {loss3:.4f}")

# -------------------------------------
# Graficar el proceso de entrenamiento
# -------------------------------------

# 1. Gráfica de pérdida (loss)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# 2. Gráfica de precisión (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión Validación')
plt.title('Precisión durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.show()

# -----------------------------------------------
# Guardar el modelo para su uso posterior
# -----------------------------------------------
model3.save("mlp_dropout_model.keras")

print("\nResultados del Modelo:")
print(f"Precisión: {acc3:.2f}, Pérdida: {loss3:.2f}")

