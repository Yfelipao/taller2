# Clasificación de Imágenes con AlexNet (CIFAR-10)

## Descripción del Proyecto

Este proyecto implementa una red neuronal convolucional basada en la arquitectura AlexNet para la clasificación de imágenes del conjunto de datos CIFAR-10.

El objetivo es entrenar un modelo capaz de reconocer 10 clases diferentes de objetos (como aviones, automóviles, aves, gatos, etc.) a partir de imágenes de tamaño 32x32 píxeles.

---

## Estructura del Proyecto

```
taller2/
├── src/
│   ├── model.py        # Definición de la arquitectura AlexNet
│   ├── train.py        # Entrenamiento del modelo
│   ├── evaluate.py     # Evaluación del modelo
│
├── models/
│   └── alexnet.h5      # Modelo entrenado
│
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
├── requirements.txt    # Dependencias del proyecto
└── README.md           # Documentación
```

---

## Requisitos

Instalar las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

---

## Uso del Proyecto

### 1. Entrenamiento del modelo

```bash
python -m src.train
```

Esto:

* Descarga el dataset CIFAR-10
* Entrena el modelo AlexNet
* Guarda el modelo en `models/alexnet.h5`

---

### 2. Evaluación del modelo

```bash
python -m src.evaluate
```

Esto:

* Carga el modelo entrenado
* Evalúa su desempeño en el conjunto de prueba
* Genera resultados en la carpeta `results/`

---

## Resultados

Los resultados generados incluyen:

* Accuracy y Loss en el conjunto de prueba
* Matriz de confusión (`confusion_matrix.png`)
* Reporte de clasificación (`classification_report.txt`) con:

  * Precision
  * Recall
  * F1-score por clase

---

## Descripción del Modelo

Se implementó una versión adaptada de AlexNet para imágenes pequeñas (32x32), que incluye:

* Capas convolucionales para extracción de características
* Batch Normalization para estabilidad del entrenamiento
* MaxPooling para reducción dimensional
* Capas densas para clasificación
* Dropout para evitar overfitting

---

## Dataset

Se utilizó el dataset CIFAR-10, que contiene:

* 60,000 imágenes a color (32x32)
* 10 clases diferentes
* División estándar:

  * 50,000 imágenes de entrenamiento
  * 10,000 imágenes de prueba

---

## Observaciones

* El modelo logra una precisión aproximada entre 70% y 80% dependiendo del entrenamiento.
* Algunas clases pueden presentar mayor confusión debido a similitudes visuales.
* El uso de regularización (Dropout) ayuda a reducir el sobreajuste.

---

## Autor

Proyecto desarrollado como parte de un taller de Visión por Computadora.

