import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

numeros = tf.keras.datasets.mnist.load_data()
#numeros = tf.keras.layers.Rescaling(scale=1 / 255)

(X_train, y_train), (X_test, y_test) = numeros
print(X_train.shape)
print(X_test.shape)

num_nombres = ["cero", "uno", "dos", "tres", "cuatro",
               "cinco", "seis", "siete", "ocho", "nueve"]

model_num = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(28, 28, 1)),
    tf.keras.layers.Rescaling(1./255), # Normalizar
    
    # Primer bloque 
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    
    # Segundo bloque 
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    # Tercer bloque
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),

    # Cuarto bloque
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    # Salida
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_num.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_num.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
prediccion = model_num.predict(np.array([X_test[0]]))
array([[4.6738582e-11, 2.7078496e-12, 4.1330747e-08, 1.3002856e-06,
        3.8394087e-11, 6.7802972e-11, 1.4370339e-16, 9.9998689e-01,
        1.2830179e-10, 1.1823602e-05]], dtype=float32)
prediccion.argmax()
predi = prediccion.argmax()
num_nombres[predi]
print(y_test[0])
plt.imshow(X_test[0],cmap='gray')
#Dropout

model_num1 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(28, 28, 1)),
    tf.keras.layers.Rescaling(1./255),  # Normaliza los valores de píxeles entre 0 y 1

    # Primer bloque 
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),

    # Segundo bloque 
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.25),  # Apaga el 25% de las neuronas después del primer pooling

    # Tercer bloque
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),

    # Cuarto bloque
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(10, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.25),  # Otro dropout tras el segundo pooling

    # Capa de salida
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),  # Dropout más fuerte antes de la capa final
    tf.keras.layers.Dense(10, activation='softmax')
])

model_num1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_num1.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
