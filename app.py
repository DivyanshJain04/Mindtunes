import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

# Load all 4 models using joblib
models = {
    "physical_health": joblib.load("Social_Media_Impact_on_Physical_Health.pkl"),
    "social_interactions": joblib.load("Social_Media_Impact_on_Social_Interactions.pkl"),
    "sleep_quality": joblib.load("Social_Media_Impact_on_Sleep_Quality.pkl"),
    "academic_performance": joblib.load("Social_Media_Impact_on_Academic_Work_Performance.pkl"),
}

@app.route('/')
def home():
    return "ML Models API is running!"

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    try:
        data = request.json  
        features = np.array(data["features"]).reshape(1, -1)  

        if model_name not in models:
            return jsonify({"error": "Invalid model name. Choose from: physical_health, social_interactions, sleep_quality, academic_performance"}), 400
        
        prediction = models[model_name].predict(features)  
        return jsonify({"model": model_name, "prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
