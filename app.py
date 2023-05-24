from flask import Flask, request
import json
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

def summary(body, status_code):
    return app.response_class(response=json.dumps(body),
                              status=status_code,
                              mimetype='application/json')


@app.route("/api-prediction/get-predict")
def hello_world():
    try:
        params = {k: frontend_fixers[k](request.args.get(k, "")) for k in frontend_fixers}
    except Exception as e:
        return summary("Brak wszystkich danych", 420)
    
    #print(f"\n\n{len(params)}\n\n")
    #print(f"\n\n{params}\n\n")

    for key in params:
        if params[key] == '':
            return summary("Brak wszystkich danych", 414)
    
    if params["year"] < 1900:
        return summary("Za niska wartość year", 415)
    
    if params["year"] > 2023:
        return summary("Za wysoka wartość year", 416)

    if params["power"] > 1000:
        return summary("Za wysoka wartość power", 417)
    
    if params["engineDisplacement"] > 20:
        return summary("Za wysoka wartość engine", 418)

    return summary(str(predict(params)), 200)


if __name__ == '__main__':
    app.run(debug=True, port=8090)