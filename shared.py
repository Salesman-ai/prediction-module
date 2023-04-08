#!/usr/bin/env python3
import json
import os

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

    return lookup


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
    "fuelType",
]


fixers = {
    "brand": encode_ordinal(readlines("brandnames")),
    "bodyType": encode_ordinal(readlines("bodynames")),
    "name": encode_ordinal(readlines("modelnames200")),
    "tranny": encode_ordinal(readlines("trannies")),
    "fuelType": encode_ordinal(readlines("fuels")),
}


def dataframize(d):
    return pd.DataFrame.from_dict({i: [d[i]] for i in d})


def fixup(ds):
    for i in fixers:
        if i in used_columns:
            ds[i] = ds[i].map(fixers[i])

    ds = ds.reindex(sorted(ds.columns), axis=1)
    print(ds)
    return ds


def load_dictionary(d):
    ds = dataframize(d)
    return fixup(ds)


def read_displacement(x):
    try:
        return float(re.sub("LTR", "", x))
    except:
        return None


def load_csv(fn, mapping_file="basic_mapper.json"):
    column_name_dictionary = json.load(
        open(os.path.join("column_mappers", mapping_file))
    )
    tmp_column_names: [str] = [column_name_dictionary[name] for name in used_columns]

    ds = pd.read_csv(
        fn,
        names=column_names,
        sep=",",
        skipinitialspace=True,
        quotechar='"',
        converters={"engineDisplacement": read_displacement},
        usecols=tmp_column_names,
        on_bad_lines="skip",
        skiprows=[0],
    )
    ds = ds.dropna()
    ds = ds.reset_index(drop=True)
    ds = fixup(ds)
    return ds
