from flask import Flask, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    value = float(request.form["value"])
    result = do_something({"value": value})
    return str(result)


def do_something(x):
    print("number got: ", x)
    return x["value"] * 2


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="127.0.0.1")
