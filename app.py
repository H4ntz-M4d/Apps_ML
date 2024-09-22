import os
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

with open('dt_model.pkl', 'rb') as model_file:
    dt_model = pickle.load(model_file)

ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/model")
def model_page():
    return render_template("model.html")

@app.route("/prediction-form")
def prediction_page():
    return render_template("Prediction/prediction_page.html")

@app.route("/prediction-upload")
def prediction_upload():
    return render_template("Prediction/prediction_upload.html")


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
        prediction = dt_model.predict(input_data)
        result = "Normal" if prediction[0] == 1 else "Berpotensi Gagal"

        return jsonify({"prediction": result, "success": True})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 400
    
@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)

            required_columns = ['Usia Mesin (tahun)', 'Jam Operasi', 'Suhu Operasi (°C)', 'Vibrasi (Hz)', 'Tingkat Kebisingan (dB)']
            
            if not all(column in df.columns for column in required_columns):
                return jsonify({"success": False, "error": "Missing required columns in the uploaded file"}), 400
            features = df[required_columns]
            predictions = dt_model.predict(features)
            os.remove(file_path)
            return jsonify({"success": True, "predictions": predictions.tolist()})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": False, "error": "File not allowed"}), 400


if __name__ == "__main__":
    app.run(debug=True)