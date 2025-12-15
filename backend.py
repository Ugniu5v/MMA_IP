from flask import Flask, request, send_from_directory
from flask_cors import CORS
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pickle
import os
import pandas as pd

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)
model = None
scalers = {}
encoders = {}
FEATURE_NAME_MAPPING = {
    "Access for people with reduced mobility": [
        "Access for people with reduced mobility"
    ],
    "Handicap access": ["Handicap access"],
    "Alarm System": ["Alarm System Security"],
    "Domotics": ["Domotics"],
    "Electric Blinds": ["Electric Blinds Security"],
    "Entry Phone": ["Entry Phone Security"],
    "Gated Complex": ["Gated Complex Security"],
    "Safe": ["Safe Security"],
    "Satellite TV": ["Satellite TV"],
    "Beachfront": ["Beachfront"],
    "Beachside": ["Beachside"],
    "Close To Forest": ["Close To Forest"],
    "Close To Marina": ["Close To Marina"],
    "Close To Sea": ["Close To Sea"],
    "Close To Skiing": ["Close To Skiing"],
    "Close to Golf": ["Close to Golf"],
    "Close to Schools": ["Close to Schools"],
    "Close to Shops": ["Close to Shops"],
    "Close to Town": ["Close to Town"],
    "Close to port": ["Close to port"],
    "Near Church": ["Near Church"],
    "Near Mosque": ["Near Mosque"],
    "Near Transport": ["Near Transport"],
    "Restaurant On Site": ["Restaurant On Site"],
    "Commercial Area": ["Commercial Area"],
    "Car Hire Facility": ["Car Hire Facility"],
    "Courtesy Bus": ["Courtesy Bus"],
    "Day Care": ["Day Care"],
    "Suburban": ["Suburban"],
    "Village": ["Village"],
    "Urbanisation": ["Urbanisation"],
    "Port": ["Port", "Port Views"],
    "Marina": ["Marina"],
    "Town": ["Town"],
    "Air Conditioning": ["Air Conditioning Climate Control"],
    "Cold A/C": ["Cold A/C Climate Control"],
    "Hot A/C": ["Hot A/C Climate Control"],
    "Pre Installed A/C": ["Pre Installed A/C Climate Control"],
    "Fireplace": ["Fireplace Climate Control"],
    "U/F Heating": ["U/F Heating Climate Control"],
    "Central Heating": ["Central Heating Climate Control"],
    "Climate Control": [],
    "Communal": ["Communal Parking"],
    "Covered": ["Covered Parking"],
    "Garage": ["Garage Parking"],
    "Underground": ["Underground Parking"],
    "Street": ["Street Parking"],
    "Private": ["Private Parking"],
    "Open": ["Open Parking"],
    "More Than One": ["More Than One Parking"],
    "Excellent": ["Excellent Condition"],
    "Good": ["Good Condition"],
    "Fair": ["Fair Condition"],
    "Recently Refurbished": ["Recently Refurbished Condition"],
    "Recently Renovated": ["Recently Renovated Condition"],
    "New Construction": ["New Construction Condition"],
    "Renovation Required": ["Renovation Required Condition"],
    "Restoration Required": ["Restoration Required Condition"],
    "Beach": ["Beach Views"],
    "Courtyard": ["Courtyard Views"],
    "Country": ["Country", "Country Views"],
    "Forest": ["Forest Views"],
    "Garden": ["Garden Views"],
    "Golf": ["Golf", "Golf Views"],
    "Lake": ["Lake Views"],
    "Mountain": ["Mountain Views"],
    "Panoramic": ["Panoramic Views"],
    "Pool": ["Pool Views"],
    "Sea": ["Sea Views"],
    "Ski": ["Ski Views"],
    "Street": ["Street Parking", "Street Views"],
    "Urban": ["Urban Views"],
    "North": ["North Orientation"],
    "North East": ["North East Orientation"],
    "North West": ["North West Orientation"],
    "East": ["East Orientation"],
    "West": ["West Orientation"],
    "South East": ["South East Orientation"],
    "South": ["South Orientation"],
    "South West": ["South West Orientation"],
    "Not Fitted": ["Not Fitted Kitchen"],
    "Partially Fitted": ["Partially Fitted Kitchen"],
    "Fully Fitted": ["Fully Fitted Kitchen"],
    "Kitchen-Lounge": ["Kitchen-Lounge Kitchen"],
    "Electricity": ["Electricity Utilities"],
    "Gas": ["Gas Utilities"],
    "Drinkable Water": ["Drinkable Water Utilities"],
    "Telephone": ["Telephone Utilities"],
    "Fiber Optic": ["Fiber Optic"],
    "WiFi": ["WiFi"],
    "Photovoltaic solar panels": ["Photovoltaic solar panels Utilities"],
    "Solar water heating": ["Solar water heating Utilities"],
    "Fully Furnished": ["Fully Furnished Furniture"],
    "Not Furnished": ["Not Furnished Furniture"],
    "Part Furnished": ["Part Furnished Furniture"],
    "Optional Furniture": ["Optional Furniture"],
    "Bar": ["Bar"],
    "Barbeque": ["Barbeque"],
    "Bargain": ["Bargain"],
    "Basement": ["Basement"],
    "Cheap": ["Cheap"],
    "Communal Garden": ["Communal Garden"],
    "Communal Pool": ["Communal Pool"],
    "Children's Pool": ["Children`s Pool Pool"],
    "Covered Terrace": ["Covered Terrace"],
    "Distressed": ["Distressed"],
    "Double Glazing": ["Double Glazing"],
    "Easy Maintenance Garden": ["Easy Maintenance Garden"],
    "Ensuite Bathroom": ["Ensuite Bathroom"],
    "Fitted Wardrobes": ["Fitted Wardrobes"],
    "Front Line Beach Complex": ["Front Line Beach Complex"],
    "Frontline Golf": ["Frontline Golf"],
    "Games Room": ["Games Room"],
    "Guest Apartment": ["Guest Apartment"],
    "Guest House": ["Guest House"],
    "Gym": ["Gym"],
    "Heated Pool": ["Heated pool"],
    "Holiday Homes": ["Holiday Homes"],
    "Indoor Pool": ["Indoor Pool"],
    "Investment": ["Investment"],
    "Jacuzzi": ["Jacuzzi"],
    "Landscaped Garden": ["Landscaped Garden"],
    "Lift": ["Lift"],
    "Luxury": ["Luxury"],
    "Marble Flooring": ["Marble Flooring"],
    "Mountain Pueblo": ["Mountain Pueblo"],
    "Off Plan": ["Off Plan"],
    "Paddle Tennis": ["Paddle Tennis"],
    "Private Garden": ["Private Garden"],
    "Private Pool": ["Private Pool"],
    "Private Terrace": ["Private Terrace"],
    "Reduced": ["Reduced"],
    "Repossession": ["Repossession"],
    "Resale": ["Resale"],
    "Room For Pool": ["Room For Pool Pool"],
    "Sauna": ["Sauna"],
    "Solarium": ["Solarium"],
    "Stables": ["Stables"],
    "Staff Accommodation": ["Staff Accommodation"],
    "Storage Room": ["Storage Room"],
    "Tennis Court": ["Tennis Court"],
    "Utility Room": ["Utility Room"],
    "With Planning Permission": ["With Planning Permission"],
    "Wood Flooring": ["Wood Flooring"],
    "Contemporary": ["Contemporary"],
}
NUMERIC_FIELDS = [
    "bedrooms",
    "bathrooms",
    "indoor_surface_area_sqm",
    "outdoor_surface_area_sqm",
]


def train_model():
    global model
    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 5,
        "learning_rate": 0.01,
        "loss": "squared_error",
    }
    model = ensemble.GradientBoostingRegressor(**params)
    if os.path.exists("df.pkl"):
        with open("df.pkl", "rb") as f:
            df = pickle.load(f)
        if "price" not in df.columns:
            raise ValueError("Expected 'price' column in df.pkl")
        X = df.drop(columns=["price"])
        y = df["price"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=4
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance Metrics:")
        print(f"MSE (Mean Squared Error): {mse:.2f}")
        print(f"RMSE (Root Mean Squared Error): {np.sqrt(mse):.2f}")
        print(f"R2 Score: {r2:.2f}")
        
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


def build_input_dataframe(form_data):
    """Create a one-row DataFrame aligned with the trained model features."""
    if model is None or not hasattr(model, "feature_names_in_"):
        raise ValueError("Model is not loaded")

    missing_fields = []
    location = (form_data.get("location") or "").strip()
    title = (form_data.get("title") or "").strip()
    if not location:
        missing_fields.append("location")
    if not title:
        missing_fields.append("title")

    numeric_map = {
        "bedroomCount": "bedrooms",
        "bathroomCount": "bathrooms",
        "indoorArea": "indoor_surface_area_sqm",
        "outdoorArea": "outdoor_surface_area_sqm",
    }
    numeric_values = {}
    for form_key, feature_name in numeric_map.items():
        raw_value = form_data.get(form_key)
        if raw_value in (None, ""):
            missing_fields.append(form_key)
            continue
        try:
            numeric_values[feature_name] = float(raw_value)
        except ValueError:
            raise ValueError(f"Invalid numeric value for {form_key}")

    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    selected_features = set(form_data.getlist("features") or [])
    mapped_feature_columns = set()
    ignored_features = []
    for name in selected_features:
        targets = FEATURE_NAME_MAPPING.get(name, [])
        if not targets:
            ignored_features.append(name)
            continue
        mapped_feature_columns.update(targets)

    if ignored_features:
        print(f"Ignored unmapped features: {ignored_features}")

    def _coerce_known_category(col: str, value: str):
        """Return value if encoder knows it, otherwise fall back to first class."""
        encoder = encoders.get(col)
        if encoder is None:
            return value
        if value in encoder.classes_:
            return value
        print(f"Unknown category for {col}: {value!r}. Falling back to {encoder.classes_[0]!r}")
        return encoder.classes_[0]

    row = {}
    for col in model.feature_names_in_:
        if col == "location":
            row[col] = _coerce_known_category(col, location)
        elif col == "title":
            row[col] = _coerce_known_category(col, title)
        elif col in NUMERIC_FIELDS:
            row[col] = numeric_values[col]
        else:
            default_value = encoders.get(col).classes_[0] if col in encoders else 0
            row[col] = "Y" if col in mapped_feature_columns else default_value

    return pd.DataFrame([row], columns=model.feature_names_in_)


@app.route("/")
def index():
    load_model()
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    load_model()
    load_encoders()
    load_scalers()
    try:
        features_df = build_input_dataframe(request.form)
        result = do_something(features_df)
        formatted_result = f"{result:,.2f}"
        return str(formatted_result)
    except ValueError as exc:
        return str(exc), 400


def do_something(x):
    global model
    global scalers
    global encoders

    if hasattr(model, "feature_names_in_"):
        x = x[model.feature_names_in_]

    # Encode categorical values
    for col, encoder in encoders.items():
        if col in x.columns:
            x[col] = encoder.transform(x[col].astype(str))

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

@app.route("/locations", methods=["GET"])
def get_locations():
    if not os.path.exists("encoders.pkl"):
        return []

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    if "location" not in encoders:
        return []

    locations = encoders["location"].classes_.tolist()
    locations.sort()

    return locations

@app.route("/titles", methods=["GET"])
def get_titles():
    if not os.path.exists("encoders.pkl"):
        return []

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    if "title" not in encoders:
        return []

    titles = encoders["title"].classes_.tolist()
    titles.sort()

    return titles

if __name__ == "__main__":
    app.run(debug=True, port=8000, host="127.0.0.1")
