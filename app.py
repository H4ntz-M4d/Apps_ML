import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from werkzeug.utils import secure_filename

app = Flask(__name__)

with open('knn_model.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

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

    df = pd.read_csv("Mesin_Data.csv")

    total_data = len(df)

    def outliers_mean(df, column_name):
        while True:
            Q1 = df[column_name].quantile(0.25)
            Q3 = df[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[
                (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
            ]
            if len(outliers) == 0:
                break
            mean = df[column_name].mean()
            df.loc[outliers.index, column_name] = mean
        return df

    df = outliers_mean(df, "Jam Operasi")

    # Set X and Y
    X = df.drop(columns=["ID Mesin", "Kegagalan"], axis=1)
    y = df["Kegagalan"].map({"TIDAK": 0, "YA": 1})

    failure_counts = df["Kegagalan"].value_counts().to_dict()

    failure_rate = {
        "YA": failure_counts.get("YA", 0),
        "TIDAK": failure_counts.get("TIDAK", 0),
    }

    random_state_accuracies = []
    best_accuracy = 0
    best_random_state = None

    for random_state in range(1, 101):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        random_state_accuracies.append((random_state, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_random_state = random_state

    random_states = [x[0] for x in random_state_accuracies]
    accuracies = [x[1] for x in random_state_accuracies]

    best_accuracy_persen = int(best_accuracy * 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=best_random_state, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    report = classification_report(y_test, y_pred)

    return render_template(
        "model.html",
        total_data=total_data,
        random_states=random_states,
        failure_rate=failure_rate,
        accuracies=accuracies,
        report=report,
        best_random_state=best_random_state,
        best_accuracy=best_accuracy_persen,
    )

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        df_uploaded = pd.read_csv(filepath)
        
        df_existing = pd.read_csv("Mesin_Data.csv")
        
        df_combined = pd.concat([df_existing, df_uploaded], ignore_index=True)
        
        df_combined.to_csv("Mesin_Data.csv", index=False)

        return redirect(url_for('model_page'))

    return "Invalid file format, please upload a CSV file.", 400

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
        prediction = knn_model.predict(input_data)
        result = "Ya" if prediction[0] == 1 else "Tidak"

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
            predictions = knn_model.predict(features)
            os.remove(file_path)
            return jsonify({"success": True, "predictions": predictions.tolist()})

        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": False, "error": "File not allowed"}), 400


if __name__ == "__main__":
    app.run(debug=True)