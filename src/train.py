from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from src.model import build_alexnet

import os

def load_data():
    # Cargar CIFAR-10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalizar imágenes (0-255 → 0-1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # One-hot encoding de etiquetas
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def train_model(epochs=10, batch_size=64):
    # Cargar datos
    x_train, y_train, x_test, y_test = load_data()

    # Crear modelo
    model = build_alexnet()

    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenamiento
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test)
    )

    # Crear carpeta models si no existe
    os.makedirs("models", exist_ok=True)

    # Guardar modelo
    model.save("models/alexnet.h5")

    return model, history


if __name__ == "__main__":
    model, history = train_model()