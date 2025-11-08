# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Motor Health API is running 🚀"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Get data from request
        temperature = float(request.args.get('temperature') or request.form.get('temperature'))
        humidity = float(request.args.get('humidity') or request.form.get('humidity'))

        # Predict
        prediction = model.predict(np.array([[temperature, humidity]]))[0]

        return jsonify({
            'temperature': temperature,
            'humidity': humidity,
            'health_score': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
