#!/usr/bin/env python3
import tensorflow.keras as k

import shared as s

model = k.models.load_model("last_model")


def predict(d):
    data = s.load_dictionary(d)
    predictions = model.predict(s.to_map(data)).flatten()
    return predictions[0]


# brand,name,bodyType,color,fuelType,year,mileage,transmission,power,price,vehicleConfiguration,engineName,engineDisplacement,date,location,link,parse_date
# Toyota,Wish,minivan,black,Gasoline,2010.0,158000.0,CVT,144.0,900000,1.8 S,2ZR-FAE,1.8 LTR,2022-09-08 00:00:00,Vladivostok,https://vladivostok.drom.ru/toyota/wish/47979764.html,2022-09-09 06:00:00
# print(
#     "real: 90000, predicted:",
#     predict(
#         {
#             "mileage": 158000,
#             "year": 2010,
#             "bodyType": "minivan",
#             "fuelType": "Gasoline",
#             "brand": "Toyota",
#             "name": "Wish",
#             "tranny": "CVT",
#             "engineDisplacement": 1.8,
#             "power": 144,
#         }
#     ),
# )
