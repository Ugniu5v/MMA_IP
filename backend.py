from flask import Flask, request, send_from_directory
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import pickle
import os
import pandas as pd

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
model = None
scalers = {}
encoders = {}


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

def load_encoders():
    global encoders
    try:
        if os.path.exists("encoders.pkl"):
            with open("encoders.pkl", "rb") as f:
                encoders = pickle.load(f)
    except FileNotFoundError:
        print("encoders.pkl is missing")


def load_scalers():
    global scalers
    try:
        if os.path.exists("scalers.pkl"):
            with open("scalers.pkl", "rb") as f:
                scalers = pickle.load(f)
    except FileNotFoundError:
        print("scalers.pkl is missing")


@app.route("/")
def index():
    load_model()
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    load_encoders()
    load_scalers()
    # Empty field value: None

    # selectedFeatures = request.form.getlist("features")
    # print(f"Selected features: {selectedFeatures}")
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

    formatted_result = f"{result:,.2f}"
    return str(formatted_result)


def do_something(x):
    global model
    global scalers
    global encoders

    # Encode categorical values
    for col, encoder in encoders.items():
        if col in x.columns:
            x[col] = encoder.transform(x[[col]])

    # Scale continues inputs
    for col, scaler in scalers.items():
        if col in x.columns:
            x[col] = scaler.transform(x[[col]]).ravel()

    prediction = model.predict(x)

    # Convert scaled prediction back to original price
    if "price" in scalers:
        prediction = scalers["price"].inverse_transform(
            prediction.reshape(-1, 1)
        )

    return float(prediction.ravel()[0])



if __name__ == "__main__":
    app.run(debug=True, port=8000, host="127.0.0.1")
