from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def create_mlp_with_batch_norm(input_dim, output_dim):
    """Define un modelo MLP con Batch Normalization."""
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(output_dim, activation='softmax')
    ])
    return model
