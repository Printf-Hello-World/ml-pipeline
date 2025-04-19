import joblib
import os
from flask import Flask, render_template, request
import datetime
import numpy as np
from src.utils.config import ConfigReader

config = ConfigReader()
model_path = config.get("model_path_for_inference")



app = Flask(__name__)

model_path = 'saved_models/'+ model_path
with open(model_path, 'rb') as model_file:
    model = joblib.load(model_file)


image_files = [
    {'class': 'Dog', 'path': 'dog.jpg'},
    {'class': 'Cat', 'path': 'cat.jpg'}
]

features = [
    {'name': 'feature 1', 'range': (3.0, 8.0), 'value': 5}
]

predictions = []

@app.route("/", methods = ["GET", "POST"])
def home():
    inputs = []
    temp = {}
    idx = 0
    if request.method == "POST":
        for key, value in request.form.items():
            features[idx]['value'] = value
            inputs.append(float(value))
            temp[key] = value
            idx+=1
        inputs = np.array(inputs).reshape(1,-1)
        prediction = model.predict(inputs)[0]
        predictions.append({"prediction": prediction, "inputs": temp})
    return render_template("index.html", image_files = image_files, features = features, predictions = predictions)

if __name__ == "__main__":
    app.run(debug=True)