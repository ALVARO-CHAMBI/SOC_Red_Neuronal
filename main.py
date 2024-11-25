import os
import sys

def train_model():
    """Permite al usuario elegir y entrenar un modelo."""
    print("\n¿Qué modelo deseas entrenar?")
    print("1. Deep MLP")
    print("2. MLP con Batch Normalization")
    print("3. MLP con Dropout")
    choice = input("Selecciona una opción (1-3): ")

    if choice == "1":
        print("\nEntrenando modelo Deep MLP...")
        os.system("python -m training.train_deep_mlp")
    elif choice == "2":
        print("\nEntrenando modelo MLP con Batch Normalization...")
        os.system("python training/train_mlp_con_batch_norm.py")
    elif choice == "3":
        print("\nEntrenando modelo MLP con Dropout...")
        os.system("python training/train_mlp_con_dropout.py")
    else:
        print("Opción no válida. Intenta de nuevo.")

def evaluate_model():
    """Permite al usuario elegir y evaluar un modelo."""
    print("\n¿Qué modelo deseas evaluar?")
    print("1. Deep MLP")
    print("2. MLP con Batch Normalization")
    print("3. MLP con Dropout")
    choice = input("Selecciona una opción (1-3): ")

    if choice == "1":
        print("\nEvaluando modelo Deep MLP...")
        os.system("python -m evaluation/evaluate_deep_mlp")
    elif choice == "2":
        print("\nEvaluando modelo MLP con Batch Normalization...")
        os.system("python evaluation/evaluate_mlp_con_batch_norm.py")
    elif choice == "3":
        print("\nEvaluando modelo MLP con Dropout...")
        os.system("python evaluation/evaluate_mlp_con_dropout.py")
    else:
        print("Opción no válida. Intenta de nuevo.")

def main():
    """Programa principal que controla el flujo."""
    while True:
        print("\n=== Menú Principal ===")
        print("1. Entrenar un modelo")
        print("2. Evaluar un modelo")
        print("3. Salir")
        choice = input("Selecciona una opción (1-3): ")

        if choice == "1":
            train_model()
        elif choice == "2":
            evaluate_model()
        elif choice == "3":
            print("Saliendo del programa. ¡Adiós!")
            sys.exit(0)
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
