from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_deep_mlp(input_dim, output_dim):
    """Define un modelo Deep MLP."""
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='softmax')  # Clasificación multiclase
    ])
    return model
