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

@app.route('/predict/physical_health', methods=['POST'])
def predict_physical_health():
    return make_prediction("physical_health")

@app.route('/predict/social_interactions', methods=['POST'])
def predict_social_interactions():
    return make_prediction("social_interactions")

@app.route('/predict/sleep_quality', methods=['POST'])
def predict_sleep_quality():
    return make_prediction("sleep_quality")

@app.route('/predict/academic_performance', methods=['POST'])
def predict_academic_performance():
    return make_prediction("academic_performance")

def make_prediction(model_name):
    try:
        data = request.json  
        features = np.array(data["features"]).reshape(1, -1)  

        if model_name not in models:
            return jsonify({"error": "Invalid model name"}), 400
        
        prediction = models[model_name].predict(features)  
        return jsonify({"model": model_name, "prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
