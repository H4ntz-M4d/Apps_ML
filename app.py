from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/model")
def model_page():
    return render_template("model.html")