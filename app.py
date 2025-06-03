from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize Flask server
app = Flask(__name__)

# Load trained components
model = joblib.load('model_v2.pkl')
scaler = joblib.load('scaler_v2.pkl')
feature_order = joblib.load('features_v2.pkl')

@app.route('/')
def index():
    return "🚀 API is active! Use POST method on /predict to get churn prediction."

@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        # Parse incoming JSON
        input_data = request.json
        df_input = pd.DataFrame([input_data])

        # Align features with training order
        df_input = df_input.reindex(columns=feature_order, fill_value=0)

        # Transform data
        transformed = scaler.transform(df_input)
        result = model.predict(transformed)[0]

        return jsonify({'prediction': str(result)})
    
    except Exception as err:
        return jsonify({'error': str(err)})

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)