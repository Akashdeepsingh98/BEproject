
import tensorflow as tf
import numpy as np
logdir="logboard"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1
    return results

dataset = tf.keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train = vectorize(x_train)
y_train = vectorize(y_train)
        
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(input_shape=(10000))
        ,tf.keras.layers.Dropout(0.3,noise_shape=None, seed=None)
        ,tf.keras.layers.Dense(50, activation=tf.nn.relu)
        ,tf.keras.layers.Dropout(0.2,noise_shape=None, seed=None)
        ,tf.keras.layers.Dense(50, activation=tf.nn.relu)
        ,tf.keras.layers.Dense(1, activation=tf.nn.softmax)])
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
model.evaluate(x_test,  y_test, verbose=2)
model.save('returnfile.h5')