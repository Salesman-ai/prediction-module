from predict import predict

ALLOWED_ERROR = 0.2


def minimal_boundary(expected: float):
    return expected * (1 - ALLOWED_ERROR)


def maximal_boundary(expected: float):
    return expected * (1 + ALLOWED_ERROR)


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
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_cheap_car():
    input = {
        "mileage": 200000,
        "year": 1989,
        "bodyType": "sedan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Corolla",
        "tranny": "AT",
        "engineDisplacement": 1.5,
        "power": 94,
    }
    expected = 18000
    output = predict(input)
    assert 10000 < output
    assert output < 30000


def test_prediction_expensive_car():
    input = {
        "mileage": 8000,
        "year": 2018,
        "bodyType": "jeep 5 doors",
        "fuelType": "Gasoline",
        "brand": "Mercedes-Benz",
        "name": "G-Class",
        "tranny": "AT",
        "engineDisplacement": 4.0,
        "power": 585,
    }
    expected = 41500000
    output = predict(input)
    assert 20000000 < output
    assert output < 60000000


def test_prediction_average_car():
    input = {
        "mileage": 158000,
        "year": 2011,
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
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_high_milage_car():
    input = {
        "mileage": 500000,
        "year": 2010,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 1.8,
        "power": 144,
    }
    expected = 600000
    output = predict(input)
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_low_milage_car():
    input = {
        "mileage": 50000,
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
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_low_year_car():
    input = {
        "mileage": 158000,
        "year": 2000,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 1.8,
        "power": 144,
    }
    expected = 500000
    output = predict(input)
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_high_year_car():
    input = {
        "mileage": 158000,
        "year": 2020,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 1.8,
        "power": 144,
    }
    expected = 1800000
    output = predict(input)
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_high_power_car():
    input = {
        "mileage": 158000,
        "year": 2010,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 1.8,
        "power": 500,
    }
    expected = 1400000
    output = predict(input)
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_low_power_car():
    input = {
        "mileage": 158000,
        "year": 2010,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 1.8,
        "power": 80,
    }
    expected = 900000
    output = predict(input)
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_low_engine_displacement_car():
    input = {
        "mileage": 158000,
        "year": 2010,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 1.0,
        "power": 144,
    }
    expected = 900000
    output = predict(input)
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)


def test_prediction_high_engine_displacement_car():
    input = {
        "mileage": 158000,
        "year": 2010,
        "bodyType": "minivan",
        "fuelType": "Gasoline",
        "brand": "Toyota",
        "name": "Wish",
        "tranny": "CVT",
        "engineDisplacement": 4.0,
        "power": 144,
    }
    expected = 1200000
    output = predict(input)
    assert minimal_boundary(expected) < output
    assert output < maximal_boundary(expected)
