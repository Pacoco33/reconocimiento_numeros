# Reconocimiento de Dígitos con CNN (MNIST)

Este proyecto implementa una red neuronal convolucional (CNN) para el reconocimiento automático de dígitos manuscritos utilizando el dataset **MNIST**.

## Descripción

El objetivo es clasificar correctamente imágenes de dígitos del 0 al 9 usando redes neuronales convolucionales (CNN), tanto con como sin capas de regularización por **Dropout**, comparando el rendimiento de ambos enfoques.

## Dataset

- **MNIST**: conjunto de datos clásico con 60,000 imágenes de entrenamiento y 10,000 de prueba.
- Cada imagen es de 28x28 píxeles, en escala de grises.

## Arquitectura del modelo (sin Dropout)

- Entrada: 28x28x1 (imagen en escala de grises)
- 4 bloques de convolución con activación ReLU
- Pooling intermedio (reducción de dimensiones)
- Capa `Flatten` + capa de salida con activación `softmax`

## Modelo mejorado (con Dropout)

Se añade Dropout después de los bloques de convolución y antes de la capa de salida para reducir el sobreajuste:

- Dropout del 25% después de cada `MaxPooling2D`
- Dropout del 50% antes de la capa `Dense` final

## Entrenamiento

- Optimización: `adam`
- Pérdida: `sparse_categorical_crossentropy`
- Métrica: `accuracy`
- Épocas: 5

## Resultados

- Ambos modelos alcanzan altas tasas de precisión en test.
- El modelo con **Dropout** ofrece mejor generalización y menor sobreajuste.

## Visualización

El script muestra una imagen del conjunto de test y predice su clase, imprimiendo las probabilidades y el nombre del número predicho.


