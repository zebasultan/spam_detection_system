"""Web prediction interface.  Run:  python app.py  ->  http://127.0.0.1:5000"""
from flask import Flask, jsonify, render_template, request

from src.predict import predict_message

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify(predict_message(data.get("message", "")))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503


if __name__ == "__main__":
    app.run(debug=False)
