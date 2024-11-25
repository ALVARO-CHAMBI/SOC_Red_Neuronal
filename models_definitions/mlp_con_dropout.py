from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_mlp_with_dropout(input_dim, output_dim):
    """Define un modelo MLP con Dropout."""
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    return model
