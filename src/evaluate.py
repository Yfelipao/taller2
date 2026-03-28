from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    # Cargar datos
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalizar
    x_test = x_test / 255.0

    # One-hot
    y_test_cat = to_categorical(y_test, 10)

    return x_test, y_test, y_test_cat


def evaluate_model():
    # Cargar modelo
    model = load_model("models/alexnet.h5")

    # Cargar datos
    x_test, y_test, y_test_cat = load_data()

    # Evaluación
    loss, accuracy = model.evaluate(x_test, y_test_cat)

    print(f"\nAccuracy en test: {accuracy:.4f}")
    print(f"Loss en test: {loss:.4f}")

    # Predicciones
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_classes)

    # Crear carpeta results
    os.makedirs("results", exist_ok=True)

    # Guardar matriz como imagen
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    plt.savefig("results/confusion_matrix.png")
    plt.close()

    # Reporte de clasificación
    report = classification_report(y_test, y_pred_classes)

    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    print("\nReporte de clasificación:\n")
    print(report)


if __name__ == "__main__":
    evaluate_model()