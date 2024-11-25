import os
import numpy as np
import tensorflow as tf
import re
import pandas as pd

# Ruta del modelo y datos
model_path = "models/deep_mlp.keras"  # Ruta del modelo en formato Keras
sheet_csv_path = "data/data.csv"

# Cargar el modelo entrenado
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el modelo en la ruta: {model_path}")
model = tf.keras.models.load_model(model_path)

# Cargar las dimensiones de las láminas desde el CSV
def load_sheet_dimensions(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {csv_path}")
    df = pd.read_csv(csv_path)
    sheet_dimensions = {}
    for _, row in df.iterrows():
        sheet_dimensions[int(row["ID"])] = {
            "Ancho": row["Ancho"],
            "Alto": row["Alto"],
            "Área Total": row["Area Total cm2"]
        }
    return sheet_dimensions

sheet_dimensions = load_sheet_dimensions(sheet_csv_path)

# Función para procesar la lista de cortes
def process_cut_list(cut_list):
    if not cut_list.strip():
        print("La lista de cortes está vacía, ingresa una lista válida.")
        return np.zeros(10), 0  # Supongamos que hay 10 tipos posibles de cortes

    pattern = r"\((\d+)\*(\d+\.?\d*)\)\*(\d+)"
    matches = re.findall(pattern, cut_list)

    if not matches:
        print("La lista de cortes no tiene un formato válido.")
        return np.zeros(10), 0

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

# Función para mostrar resultados
def display_results(best_sheet_id, total_area_used):
    """Muestra las medidas y detalles de la lámina seleccionada."""
    dimensions = sheet_dimensions.get(best_sheet_id, {})
    if dimensions:
        sheet_area = dimensions["Área Total"]
        leftover_area = sheet_area - total_area_used
        used_percentage = (total_area_used / sheet_area) * 100
        leftover_percentage = 100 - used_percentage

        print(f"\nLámina óptima seleccionada:")
        print(f"  ID: {best_sheet_id}")
        print(f"  Medidas:")
        print(f"    Ancho: {dimensions['Ancho']} cm")
        print(f"    Alto: {dimensions['Alto']} cm")
        print(f"    Área Total: {sheet_area} cm²")
        print(f"Detalles de los cortes:")
        print(f"  Área Ocupada: {total_area_used:.2f} cm²")
        print(f"  Área Sobrante: {leftover_area:.2f} cm²")
        print(f"  Porcentaje de Uso: {used_percentage:.2f}%")
        print(f"  Porcentaje Sobrante: {leftover_percentage:.2f}%")
    else:
        print("No se encontraron medidas para la lámina seleccionada.")

# Función principal
# Función principal
def main():

    while True:
        print("\nOpciones:")
        print("1. Calcular la lámina óptima.")
        print("2. Salir.")
        option = input("Selecciona una opción (1/2): ")

        if option == "2":
            print("Saliendo")
            break

        elif option == "1":
            user_input = input(
                "Introduce la lista de cortes (formato: '(107*61.9)*3+(107*62.4)*2'): ")
            processed_input, total_area_used = process_cut_list(user_input)

            if total_area_used == 0:
                print("No se procesaron datos válidos. Inténtalo de nuevo.")
                continue

            processed_input = processed_input.reshape(1, -1)

            # Predecir la mejor lámina (sin agregar columnas adicionales)
            prediction = model.predict(processed_input)
            best_sheet_id = np.argmax(prediction) + 1

            # Mostrar resultados
            display_results(best_sheet_id, total_area_used)
        else:
            print("Opción no válida, selecciona 1 o 2.")


if __name__ == "__main__":
    main()
