from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)
@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/model")
def model_page():
    return render_template("model.html")

@app.route("/prediction-form")
def prediction_page():
    return render_template("Prediction/prediction_page.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        input_data = pd.DataFrame({
            'Usia Mesin (tahun)': [float(data['usia_mesin'])],
            'Jam Operasi': [float(data['jam_operasi'])],
            'Suhu Operasi (°C)': [float(data['suhu_operasi'])],
            'Vibrasi (Hz)': [float(data['vibrasi'])],
            'Tingkat Kebisingan (dB)': [float(data['tingkat_kebisingan'])]
        })
        prediction = knn_model.predict(input_data)
        result = "YA" if prediction[0] == 1 else "TIDAK"

        return jsonify({"prediction": result, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400
    
if __name__ == "__main__":
    app.run(debug=True)