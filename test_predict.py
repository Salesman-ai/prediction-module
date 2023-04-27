from predict import predict


def test_prediction():
    input = {
        "mileage": 158000,
        "year": 2010,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 1.8,
        "power": 144,
    }
    expected = 900000
    output = predict(input)
    assert output > (expected * 0.5)
    assert output < (expected * 2.0)
