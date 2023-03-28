#!/usr/bin/env python3
import tensorflow.keras as k

import shared as s

model = k.models.load_model("last_model")


def predict(d):
    data = s.load_dictionary(d)
    predictions = model.predict(data).flatten()
    return predictions[0]


print(
    predict(
        {
            "mileage": 8000,
            "year": 2000,
            "bodyType": "open",
            "brand": "fiat",
            "name": "124 Spider",
            "tranny": "automatic",
            "engineDisplacement": 1.9,
            "power": 150,
        }
    )
)
