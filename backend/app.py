from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lead_model.pkl")

# Load model
model = joblib.load(MODEL_PATH)


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    engagement = float(data["engagement"])
    size = float(data["company_size"])
    visits = float(data["visits"])

    result = model.predict([[engagement, size, visits]])

    if result[0] == 1:
        output = "High Quality Lead - Send Email "
    else:
        output = "Low Quality Lead - Skip "

    return jsonify({"prediction": output})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
