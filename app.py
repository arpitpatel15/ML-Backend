# app.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "IoT Health Monitoring API is running 🚀"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Get parameters
        temperature = float(request.args.get('temperature') or request.form.get('temperature'))
        humidity = float(request.args.get('humidity') or request.form.get('humidity'))
        current = float(request.args.get('current') or request.form.get('current'))
        voltage = float(request.args.get('voltage') or request.form.get('voltage'))
        vibration_total = float(request.args.get('vibration_total') or request.form.get('vibration_total'))
        rpm = float(request.args.get('rpm') or request.form.get('rpm'))

        # Create input DataFrame
        input_data = pd.DataFrame([{
            'temperature': temperature,
            'humidity': humidity,
            'current': current,
            'voltage': voltage,
            'vibration_total': vibration_total,
            'rpm': rpm
        }])

        # Predict
        prediction = model.predict(input_data)[0]

        return jsonify({
            'temperature': temperature,
            'humidity': humidity,
            'current': current,
            'voltage': voltage,
            'vibration_total': vibration_total,
            'rpm': rpm,
            'health_status': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
