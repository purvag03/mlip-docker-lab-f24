from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
import time

app = Flask(__name__)

# Check if the model file exists and wait until it does
model_path = '/app/models/iris_model.pkl'

while not os.path.exists(model_path):
    print(f"Waiting for model file at {model_path}...")
    time.sleep(5)

# Load the trained model from the shared volume
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    get_json = request.get_json()
    iris_input = get_json['input']
    input = np.array(iris_input).reshape(1, -1)
    prediction = model.predict(input)
    return jsonify({'Prediction': prediction.tolist()})

@app.route('/')
def hello():
    return 'Welcome to Docker Lab'

if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')