# reconocimiento_numeros
Modelo de reconocimiento con Keras y Tensorflow + Dropout

Esto es un modelo de aprendizaje profundo (deep learning) que aprende a reconocer imágenes de números escritos a mano.
Usa técnicas como convoluciones, max-pooling y dropout, que son comunes en tareas de visión por computadora.

#funcionamiento de las capas
Imagen (28x28) → Rescaling (valores 0-1) → Conv2D x 2 (busca patrones) → Conv2D x 2 + MaxPooling (reduce tamaño) →
[Dropout si hay] → Conv2D x 2 → Conv2D x 2 + MaxPooling → [Dropout si hay] → Flatten (aplanar) → Dropout (opcional) → Dense (10 salidas con probabilidades)
