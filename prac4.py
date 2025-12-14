# pip install matplotlib
# pip install scikit-learn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# binary classification dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

def create_mlp_relu():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_mlp_tanh():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='tanh', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


model_relu = create_mlp_relu()
history_relu = model_relu.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=0
)

model_tanh = create_mlp_tanh()
history_tanh = model_tanh.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=0
)

plt.figure(figsize=(10,5))

plt.plot(history_relu.history['loss'], label='ReLU - Train Loss')
plt.plot(history_relu.history['val_loss'], label='ReLU - Val Loss')

plt.plot(history_tanh.history['loss'], label='Tanh - Train Loss')
plt.plot(history_tanh.history['val_loss'], label='Tanh - Val Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Convergence Curve")
plt.legend()
plt.show()