import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

# -----------------------------
# Dataset
# -----------------------------
X, y = make_classification(n_samples=1000, n_features=20,
                           n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Model function
# -----------------------------
def create_model(optimizer):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(20,)),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# -----------------------------
# Optimizers
# -----------------------------
optimizers = {
    "SGD": SGD(learning_rate=0.01),
    "RMSProp": RMSprop(learning_rate=0.001),
    "Adam": Adam(learning_rate=0.001)
}

history_dict = {}

# -----------------------------
# Training
# -----------------------------
for name, opt in optimizers.items():
    print(f"\nTraining with {name}")
    model = create_model(opt)
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    history_dict[name] = history

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(10,4))
for name in history_dict:
    plt.plot(history_dict[name].history['loss'], label=name)
plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure(figsize=(10,4))
for name in history_dict:
    plt.plot(history_dict[name].history['accuracy'], label=name)
plt.title("Training Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# ---b-

from tensorflow.keras.initializers import RandomNormal, GlorotUniform, HeNormal

def create_model_initializer(initializer):
    model = Sequential([
        Dense(32, activation='relu',
              kernel_initializer=initializer, input_shape=(20,)),
        Dense(16, activation='relu',
              kernel_initializer=initializer),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

initializers = {
    "Random Normal": RandomNormal(mean=0.0, stddev=0.05),
    "Xavier": GlorotUniform(),
    "He": HeNormal()
}

history_init = {}

for name, init in initializers.items():
    print(f"\nTraining with {name} Initialization")
    model = create_model_initializer(init)
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    history_init[name] = history

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure(figsize=(10,4))
for name in history_init:
    plt.plot(history_init[name].history['loss'], label=name)
plt.title("Loss Comparison for Weight Initialization")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
