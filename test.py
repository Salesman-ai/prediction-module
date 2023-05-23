import requests

HOST = "http://127.0.0.1:5000"
PREDICTION_ENDPOINT = "get-predict"

frontend_fixers = {
    "mileage": 158000,
    "year": 2010,
    "bodyType": "minivan",
    "fuelType": "Gasoline",
    "brand": "Toyota",
    "name": "Wish",
    "tranny": "",
    "engineDisplacement": 1.8,
    "power": 0,
}

a = requests.get(f"{HOST}/api-prediction/{PREDICTION_ENDPOINT}", params=frontend_fixers)

print(a.text)
