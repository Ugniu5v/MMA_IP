from flask import Flask, request, send_from_directory
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import pickle
import os
import pandas as pd

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
model = None


def train_model():
    global model
    model = LinearRegression()
    if os.path.exists("df.pkl"):
        with open("df.pkl", "rb") as f:
            df = pickle.load(f)
        X = df[
            [
                "bedrooms",
                "bathrooms",
                "indoor_surface_area_sqm",
                "outdoor_surface_area_sqm",
            ]
        ]
        y = df["price"]
        model.fit(X, y)
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)


def load_model():
    global model

    if not os.path.exists("model.pkl"):
        # async train??
        train_model()
    else:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)


@app.route("/")
def index():
    load_model()
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    # Empty field value: None

    # Not encoded data (strings)
    # location = request.form.get("location")
    # title = request.form.get("title")
    bedroomCount = int(request.form.get("bedroomCount"))
    bathroomCount = int(request.form.get("bathroomCount"))
    indoorArea = float(request.form.get("indoorArea"))
    outdoorArea = float(request.form.get("outdoorArea"))

    result = do_something(
        pd.DataFrame.from_dict(
            {
                # "title": title,
                # "location": location,
                "bedrooms": [bedroomCount],
                "bathrooms": [bathroomCount],
                "indoor_surface_area_sqm": [indoorArea],
                "outdoor_surface_area_sqm": [outdoorArea],
            }
        )
    )

    return str(result)


def do_something(x):
    global model

    prediction = model.predict(x)
    return prediction[0]


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="127.0.0.1")
