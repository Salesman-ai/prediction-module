#!/usr/bin/env python3

#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regex as re
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.layers import IntegerLookup, Normalization, StringLookup


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


fixers = {
    "brand": encode_ordinal(readlines("brandnames")),
    "bodyType": encode_ordinal(readlines("bodynames")),
    "name": encode_ordinal(readlines("modelnames200")),
    "tranny": encode_ordinal(readlines("trannies")),
}


def weird_thing(x, ds):
    one_hot = pd.get_dummies(ds[x])
    one_hot.columns = [x + str(i) for i in one_hot.columns]
    del ds[x]
    return pd.concat([ds, one_hot], axis=1)


def dataframize(d):
    return pd.DataFrame.from_dict({i: [d[i]] for i in d})


def fixup(ds):
    for i in fixers:
        ds[i] = ds[i].map(fixers[i])
    ds = weird_thing("tranny", ds)
    ds = weird_thing("brand", ds)
    ds = weird_thing("bodyType", ds)
    ds = weird_thing("name", ds)
    return ds


model = k.models.load_model("last_model")


def predict(d):
    # data = fixup(dataframize(d))
    # predictions = model.predict(data).flatten()
    # return predictions[0]
    return 50000.0


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
predict(
    {
        "mileage": 8000,
        "year": 2000,
        "bodyType": "open",
        "brand": "fiat",
        "name": "124 Spider",
        "tranny": "automatic",
        "engineDisplacement": 1.5,
        "power": 150,
    }
)
