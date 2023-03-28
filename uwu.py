#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k

import shared as s


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [price]")
    plt.legend()
    plt.grid(True)


start = time.time()
ds = s.load_csv("r25c.csv")
# ds = s.load_csv("r25_500k.csv")
# ds = s.load_csv("r25_100k.csv")
end = time.time()
print("elapsed: ", end - start)

train_dataset = ds.sample(frac=0.8, random_state=0)
test_dataset = ds.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()
# ¯\_(ツ)_/¯
train_vals = train_features.pop("price")
test_vals = test_features.pop("price")

normalizer = k.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

model = tf.keras.Sequential(
    [
        normalizer,
        k.layers.Dense(units=300, activation="relu"),
        k.layers.Dense(units=300, activation="relu"),
        k.layers.Dense(units=100, activation="relu"),
        k.layers.Dense(units=10, activation="relu"),
        k.layers.Dense(units=1, activation="relu"),
    ]
)

model.summary()
model.compile(
    optimizer=k.optimizers.Adam(learning_rate=0.02),
    loss="mean_absolute_percentage_error",
)

history = model.fit(
    train_features,
    train_vals,
    epochs=5,
    validation_split=0.2,
)
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()


def test(m):
    test_predictions = m.predict(test_features).flatten()
    # plot_loss(history)
    # plt.show()
    errors = test_predictions - test_vals
    aerrs = [abs(x) for x in errors]

    def cpct(x, y):
        a = (x / y) * 100
        if a > 500:
            print(x, y, "-> ", a)
            return None
        return a

    pct_errs = [cpct(x, y) for (x, y) in zip(errors, test_vals)]
    pct_errs = [x for x in pct_errs if x is not None]
    plt.hist(pct_errs, bins=100)
    plt.xlabel("Prediction Error %")
    plt.ylabel("Count")
    plt.show()


model.save("last_model")
# test(model)
