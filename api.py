from flask import Flask, request

from predict import predict

app = Flask(__name__)


frontend_fixers = {
    "mileage": float,
    "year": float,
    "bodyType": str,
    "fuelType": str,
    "brand": str,
    "name": str,
    "tranny": str,
    "engineDisplacement": float,
    "power": float,
}


@app.route("/predict/")
def hello_world():
    params = {k: frontend_fixers[k](request.args.get(k, "")) for k in frontend_fixers}
    return predict(params)
