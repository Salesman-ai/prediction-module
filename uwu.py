#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
import numpy as np
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
#print("elapsed: ", end - start)

train_dataset = ds.sample(frac=0.8, random_state=0)
test_dataset = ds.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()
# ¯\_(ツ)_/¯
train_vals = train_features.pop("price")
test_vals = test_features.pop("price")

normalizer = k.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))


def make_input(x):
    size = s.columns_sizes[x]
    if size is None:
        norm = k.layers.Normalization(axis=-1)
        norm.adapt(train_features[x])
        inp = tf.keras.Input(shape=(1,), name=x)
        return [inp, norm(inp)]
    else:
        enc = tf.keras.layers.CategoryEncoding(
            num_tokens=size + 5, output_mode="one_hot", sparse=False
        )
        inp = tf.keras.Input(shape=(1,), name=x)
        return [inp, enc(inp)]


boths = [make_input(i) for i in s.used_columns if i != "price"]
preprocessed = [x[1] for x in boths]
inputs = [x[0] for x in boths]
#print("nya")

# inputs = k.layers.Concatenate()(inputs)
x = k.layers.Concatenate()(preprocessed)
x = k.layers.Dense(256, activation="relu")(x)
x = k.layers.Dropout(0.5)(x)
x = k.layers.Dense(256, activation="relu")(x)
x = k.layers.Dense(64, activation="relu")(x)
x = k.layers.Dense(8, activation="relu")(x)
x = k.layers.Dense(1, activation="relu")(x)
outputs = x
model = k.Model(inputs={x.name: x for x in inputs}, outputs=outputs, name="meow")
#print("nya")

model.summary()
model.compile(
    optimizer=k.optimizers.Adam(learning_rate=0.02),
    loss="mean_absolute_percentage_error",
)

#print("nya")

history = model.fit(
    s.to_map(train_features),
    train_vals,
    epochs=30,
    validation_split=0.2,
)


def test(m):
    def cpct(x, y):
        a = (x / y) * 100
        if a > 500:
            #print(x, y, "-> ", a)
            return None
        return a

    test_predictions = m.predict(s.to_map(test_features)).flatten()
    errors = test_predictions - test_vals
    pct_errs = [cpct(x, y) for (x, y) in zip(errors, test_vals)]
    pct_errs = [x for x in pct_errs if x is not None]
    plt.hist(pct_errs, bins=100)
    plt.xlabel("Prediction Error %")
    plt.ylabel("Count")
    plt.show()


model.save("last_model")
test(model)
