import numpy as np
import pandas as pd
import re

def process_cut_list(cut_list):
    """Convierte una lista de cortes en un vector numérico."""
    if pd.isna(cut_list) or not cut_list.strip():
        return np.zeros(10)  # Supongamos 10 tipos posibles de cortes
    pattern = r"\((\d+)\*(\d+\.?\d*)\)\*(\d+)"
    matches = re.findall(pattern, cut_list)
    vector = np.zeros(10)
    for i, match in enumerate(matches):
        cantidad = int(match[2])
        vector[i] = cantidad
    return vector

def preprocess_data(csv_path):
    """Preprocesa los datos del archivo CSV."""
    df = pd.read_csv(csv_path)
    X = np.array([process_cut_list(cuts) for cuts in df['Lista de cortes']])
    y = pd.get_dummies(df['ID']).values  # One-hot encoding para las etiquetas
    return X, y
