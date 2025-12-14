import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Dataset (intentionally easy to overfit)
X, y = make_classification(
    n_samples=800,
    n_features=20,
    n_informative=15,
    n_classes=2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# model without regularization(baseline-overfit)

def baseline_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(20,)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

baseline = baseline_model()
history_base = baseline.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    verbose=0
)


# model with L2 regularization and Dropout + EarlyStopping
def regularized_model():
    model = Sequential([
        Dense(128, activation='relu',
              kernel_regularizer=l2(0.001),
              input_shape=(20,)),
        Dropout(0.5),
        Dense(128, activation='relu',
              kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reg = regularized_model()
history_reg = reg.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=0
)

# Plot: Loss & Accuracy
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history_base.history['accuracy'], label='Train (No Reg)')
plt.plot(history_base.history['val_accuracy'], label='Val (No Reg)')
plt.plot(history_reg.history['accuracy'], label='Train (Reg)')
plt.plot(history_reg.history['val_accuracy'], label='Val (Reg)')
plt.title("Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history_base.history['loss'], label='Train (No Reg)')
plt.plot(history_base.history['val_loss'], label='Val (No Reg)')
plt.plot(history_reg.history['loss'], label='Train (Reg)')
plt.plot(history_reg.history['val_loss'], label='Val (Reg)')
plt.title("Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()


# b) Regression using Mean Squared Error (MSE)

from sklearn.datasets import make_regression
from tensorflow.keras.losses import MeanSquaredError

# dataset
X, y = make_regression(
    n_samples=1000,
    n_features=5,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model 
model = Sequential([
    Dense(64, activation='relu', input_shape=(5,)),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(
    optimizer='adam',
    loss=MeanSquaredError()
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=0

)

# Plot Regression Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Regression using MSE Loss")
plt.legend()
plt.show()

