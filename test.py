import requests

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

a = requests.get("http://127.0.0.1:5000/api-prediction/get-predict", params=frontend_fixers)

# print(a.text)