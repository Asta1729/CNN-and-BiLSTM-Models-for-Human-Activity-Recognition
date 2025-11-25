import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense


def bilstm_model(input_shape):
    model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(561, 1)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dense(6, activation="softmax")])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
    return model
