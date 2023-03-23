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

# used_columns = ["price", "mileage", "year", "bodyType", "brand", "name", "tranny"]
used_columns = [
    "price",
    "mileage",
    "year",
    "bodyType",
    "brand",
    "name",
    "tranny",
    "power",
    "engineDisplacement",
]
# used_columns = ["price", "mileage", "year", "name"]


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
    "name": encode_ordinal(readlines("modelnames200")),
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


# ds = load("r25_99k.csv")
ds = load("r25c.csv")


def weird_thing(x):
    global ds
    one_hot = pd.get_dummies(ds[x])
    one_hot.columns = [x + str(i) for i in one_hot.columns]
    del ds[x]
    ds = pd.concat([ds, one_hot], axis=1)


weird_thing("tranny")
weird_thing("brand")
weird_thing("bodyType")
weird_thing("name")
print(ds)
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
        k.layers.Dense(units=300, activation="relu"),
        k.layers.Dense(units=300, activation="relu"),
        k.layers.Dense(units=100, activation="relu"),
        k.layers.Dense(units=10, activation="relu"),
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
    epochs=50,
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


def cpct(x, y):
    a = (x / y) * 100
    if a > 500:
        print(x, y, "-> ", a)
        return None
    return a


percentage_errors = [cpct(x, y) for (x, y) in zip(errors, test_vals)]
percentage_errors = [x for x in percentage_errors if x is not None]
plt.hist(percentage_errors, bins=100)
plt.xlabel("Prediction Error %")
plt.ylabel("Count")
model.save("last_model")
plt.show()
