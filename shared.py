#!/usr/bin/env python3

import numpy as np
import pandas as pd
import regex as re


def readlines(fn):
    a = []
    with open(fn) as f:
        for li in f:
            a.append(li.rstrip())
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

    def one_hot(e):
        j = lookup(e)
        a = np.array([np.float32(i == j) for i in range(i + 2)])
        return a

    return one_hot


fixers = {
    "brand": encode_ordinal(readlines("brandnames")),
    "bodyType": encode_ordinal(readlines("bodynames")),
    "name": encode_ordinal(readlines("modelnames200")),
    "tranny": encode_ordinal(readlines("trannies")),
}


def weird_thing(x, ds):
    one_hot = pd.DataFrame(ds[x].to_list())
    one_hot.columns = [x + str(i) for i in one_hot.columns]
    del ds[x]
    ans = pd.concat([ds, one_hot], axis=1)
    return ans


def dataframize(d):
    return pd.DataFrame.from_dict({i: [d[i]] for i in d})


def fixup(ds):
    for i in fixers:
        ds[i] = ds[i].map(fixers[i])

    for i in ["tranny", "brand", "bodyType", "name"]:
        ds = weird_thing(i, ds)

    ds = ds.reindex(sorted(ds.columns), axis=1)
    return ds


def load_dictionary(d):
    ds = dataframize(d)
    return fixup(ds)


def read_displacement(x):
    try:
        return float(re.sub("LTR", "", x))
    except:
        return None


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


def load_csv(fn):
    ds = pd.read_csv(
        fn,
        names=column_names,
        sep=",",
        skipinitialspace=True,
        quotechar='"',
        converters={"engineDisplacement": read_displacement},
        usecols=used_columns,
        on_bad_lines="skip",
        skiprows=[0],
    )
    ds = ds.dropna()
    ds = ds.reset_index(drop=True)
    ds = fixup(ds)
    return ds
