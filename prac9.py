import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# -------------------------------
# Load IMDB Dataset
# -------------------------------
vocab_size = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# -------------------------------
# Model Builder Function
# -------------------------------
def build_model(rnn_type):
    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_len))

    if rnn_type == "RNN":
        model.add(SimpleRNN(64))
    elif rnn_type == "LSTM":
        model.add(LSTM(64))
    elif rnn_type == "GRU":
        model.add(GRU(64))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------
# Train Models
# -------------------------------
histories = {}
models = ["RNN", "LSTM", "GRU"]

for m in models:
    print(f"\nTraining {m} model...")
    model = build_model(m)
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )
    histories[m] = history

# -------------------------------
# Plot Accuracy
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
for m in models:
    plt.plot(histories[m].history['val_accuracy'], label=m)
plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# -------------------------------
# Plot Loss
# -------------------------------
plt.subplot(1,2,2)
for m in models:
    plt.plot(histories[m].history['val_loss'], label=m)
plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()