import tensorflow as tf
import matplotlib.pyplot as plt

keras = tf.keras
mnist = keras.datasets.mnist
models = keras.models
layers = keras.layers
    

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = models.Sequential([                    
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),                     
    layers.Dense(128, activation='relu'), 
    layers.Dense(64, activation='relu'),  
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy}')

model.save('my_model.keras')

import numpy as np 

num = 125
plt.imshow(x_test[num])
image_to_predict = x_test[num]
image_to_predict = np.expand_dims(image_to_predict, axis=0)
print("Model Prediction is : ", np.argmax(model.predict(image_to_predict)))


