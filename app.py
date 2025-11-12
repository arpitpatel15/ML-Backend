from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Global variable to store latest sensor data
latest_data = {}

@app.route('/')
def home():
    return "<h2>🚀 IoT Health Monitoring API is Live on Render!<br>Send data to /predict and view results at /dashboard</h2>"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global latest_data
    try:
        # Get parameters (from GET or POST)
        temperature = float(request.args.get('temperature') or request.form.get('temperature'))
        humidity = float(request.args.get('humidity') or request.form.get('humidity'))
        current = float(request.args.get('current') or request.form.get('current'))
        voltage = float(request.args.get('voltage') or request.form.get('voltage'))
        vibration_total = float(request.args.get('vibration_total') or request.form.get('vibration_total'))
        rpm = float(request.args.get('rpm') or request.form.get('rpm'))

        # Prepare input for prediction
        input_data = pd.DataFrame([{
            'temperature': temperature,
            'humidity': humidity,
            'current': current,
            'voltage': voltage,
            'vibration_total': vibration_total,
            'rpm': rpm
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Save latest values
        latest_data = {
            'temperature': temperature,
            'humidity': humidity,
            'current': current,
            'voltage': voltage,
            'vibration_total': vibration_total,
            'rpm': rpm,
            'health_status': prediction
        }

        # Return JSON for API users (e.g., IoT device)
        return jsonify(latest_data)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/dashboard')
def dashboard():
    """Display the latest received sensor data on a web dashboard."""
    if not latest_data:
        return "<h2>No data received yet.<br>Send a GET/POST request to /predict first.</h2>"
    return render_template('dashboard.html', data=latest_data)

if __name__ == "__main__":
    # Render uses PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
