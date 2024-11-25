import numpy as np
import tensorflow as tf
import re
import pandas as pd

# Cargar el modelo entrenado en formato nativo Keras
model_path = "mlp_dropout_model.keras"  # Ruta del modelo en formato Keras
model = tf.keras.models.load_model(model_path)

# Cargar las dimensiones de las láminas desde el CSV
def load_sheet_dimensions(csv_path):
    df = pd.read_csv(csv_path)
    sheet_dimensions = {}
    for index, row in df.iterrows():
        sheet_dimensions[int(row["ID"])] = {
            "Ancho": row["Ancho"],
            "Alto": row["Alto"],
            "Área Total": row["Area Total cm2"]
        }
    return sheet_dimensions

# Ruta del archivo CSV con información de las láminas
sheet_csv_path = "data/data.csv"
sheet_dimensions = load_sheet_dimensions(sheet_csv_path)

# Función para procesar la lista de cortes
def process_cut_list(cut_list):
    if not cut_list.strip():
        return np.zeros(10), 0  # Supongamos que hay 10 tipos posibles de cortes
    pattern = r"\((\d+)\*(\d+\.?\d*)\)\*(\d+)"
    matches = re.findall(pattern, cut_list)
    vector = np.zeros(10)
    total_area_used = 0
    for i, match in enumerate(matches):
        width = int(match[0])
        height = float(match[1])
        quantity = int(match[2])
        area = width * height * quantity
        total_area_used += area
        vector[i] = quantity
    return vector, total_area_used

# Función principal del programa
def main():
    print("Bienvenido al optimizador de láminas")
    while True:
        # Solicitar la lista de cortes al usuario
        user_input = input(
            "Introduce la lista de cortes (formato: '(107*61.9)*3+(107*62.4)*2') o 'salir' para terminar: ")
        if user_input.lower() == "salir":
            print("Saliendo del programa. ¡Adiós!")
            break

        # Procesar la entrada del usuario y calcular el área ocupada
        processed_input, total_area_used = process_cut_list(user_input)
        processed_input = processed_input.reshape(1, -1)

        # Agregar columnas de área usada y sobrante (simulando valores promedio)
        area_used_percentage = 0.95  # Suponiendo un 95% de área usada (puedes personalizar esto)
        area_left_percentage = 1 - area_used_percentage
        extra_features = np.array([[area_used_percentage, area_left_percentage]])

        # Combinar el vector de cortes con las características adicionales
        full_input = np.hstack((processed_input, extra_features))

        # Predecir la mejor lámina
        prediction = model.predict(full_input)
        best_sheet_id = np.argmax(prediction) + 1

        # Mostrar la lámina óptima
        print(f"La lámina óptima para la lista de cortes es la ID: {best_sheet_id}")

        # Buscar y mostrar las medidas de la lámina
        dimensions = sheet_dimensions.get(best_sheet_id, {})
        if dimensions:
            sheet_area = dimensions["Área Total"]
            leftover_area = sheet_area - total_area_used
            used_percentage = (total_area_used / sheet_area) * 100
            leftover_percentage = 100 - used_percentage

            print(f"Medidas de la lámina ID {best_sheet_id}:")
            print(f"  Ancho: {dimensions['Ancho']} cm")
            print(f"  Alto: {dimensions['Alto']} cm")
            print(f"  Área Total: {sheet_area} cm²")
            print(f"Detalles de los cortes:")
            print(f"  Área Ocupada: {total_area_used:.2f} cm²")
            print(f"  Área Sobrante: {leftover_area:.2f} cm²")
            print(f"  Porcentaje de Uso: {used_percentage:.2f}%")
            print(f"  Porcentaje Sobrante: {leftover_percentage:.2f}%")
        else:
            print("No se encontraron medidas para la lámina seleccionada.")

if __name__ == "__main__":
    main()
