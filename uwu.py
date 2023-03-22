#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import IntegerLookup, Normalization, StringLookup

column_names = [
    "brand",
    "name",
    "bodyType",
    "color",
    "fuelType",
    "year",
    "mileage",
    "tranny",
    "power",
    "price",
    "vehicleConfiguration",
    "engineName",
    "engineDisplacement",
    "date",
    "location",
    "link",
    "parse date",
]

used_columns = ["price", "mileage", "year", "bodyType", "brand", "name", "tranny"]


def readlines(fn):
    a = []
    with open(fn) as f:
        for l in f:
            a.append(l.rstrip())
    return a


def encode_ordinal(features):
    a = {}
    i = 0
    for x in features:
        if x not in a:
            a[x] = i
            i = i + 1

    def lookup(e):
        if e in a:
            return a[e]
        return i + 1

    return lookup


def encode_oh(features):
    a = {}
    i = 0
    for x in features:
        if x not in a:
            a[x] = i
            i = i + 1

    def lookup(e):
        if e in a:
            return a[e]
        return i + 1

    return lookup


def read_displacement(x):
    try:
        return float(re.sub("LTR", "", x))
    except:
        return None


fixers = {
    "engineDisplacement": read_displacement,
    "brand": encode_ordinal(readlines("brandnames")),
    "bodyType": encode_ordinal(readlines("bodynames")),
    "name": encode_ordinal(readlines("modelnames")),
    "tranny": encode_ordinal(readlines("trannies")),
}


def load(fn):
    raw = pd.read_csv(
        fn,
        names=column_names,
        sep=",",
        skipinitialspace=True,
        quotechar='"',
        converters=fixers,
        usecols=used_columns,
        on_bad_lines="skip",
        skiprows=[0],
    )
    return raw.dropna()


ds = load("r25_99k.csv")
# ds = load("r25c.csv")

# sns.pairplot(ds[used_columns])
# plt.show()

train_dataset = ds.sample(frac=0.8, random_state=0)
test_dataset = ds.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()
# ¯\_(ツ)_/¯
train_vals = train_features.pop("price")
test_vals = test_features.pop("price")

normalizer = k.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# price = np.array(train_features["price"])

# price_normalizer = k.layers.Normalization(
#    input_shape=[1],
#    axis=None,
# )
# price_normalizer.adapt(price)

model = tf.keras.Sequential(
    [
        normalizer,
        k.layers.Dense(units=500, activation="relu"),
        k.layers.Dense(units=100, activation="relu"),
        k.layers.Dense(units=50, activation="relu"),
        k.layers.Dense(units=1, activation="relu"),
    ]
)

model.summary()
model.compile(
    optimizer=k.optimizers.Adam(learning_rate=0.03),
    loss="mean_absolute_percentage_error",
)

# %time

history = model.fit(
    train_features,
    train_vals,
    epochs=5,
    # Suppress logging.
    verbose=1,
    # Calculate validation results on 20% of the training data.
    validation_split=0.2,
)
hist = pd.DataFrame(history.history)
hist["epoch"] = history.epoch
hist.tail()


def plot_loss(history):
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    # plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [price]")
    plt.legend()
    plt.grid(True)


test_predictions = model.predict(test_features).flatten()
# plot_loss(history)
# plt.show()
errors = test_predictions - test_vals
aerrs = [abs(x) for x in errors]
percentage_errors = [(x / y) * 100 for (x, y) in zip(errors, test_vals)]
plt.hist(percentage_errors, bins=100)
plt.xlabel("Prediction Error %")
plt.ylabel("Count")
model.save("last_model")
plt.show()
