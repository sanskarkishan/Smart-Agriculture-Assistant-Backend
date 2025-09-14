from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables to store models and scalers
crop_model = None
crop_scaler = None
fertilizer_model = None
fertilizer_scaler = None
soil_encoder = None
crop_encoder = None

# Update dataset paths to point inside the backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CROP_DATASET_PATH = os.path.join(BASE_DIR, 'Dataset', 'Crop_recommendation.csv')
FERTILIZER_DATASET_PATH = os.path.join(BASE_DIR, 'Dataset', 'Fertilizer Prediction.csv')

def load_crop_model():
    """Load and train the crop recommendation model"""
    global crop_model, crop_scaler
    
    try:
        # Load the dataset
        crop_data = pd.read_csv(CROP_DATASET_PATH)
        
        # Prepare features and target
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        X = crop_data[features]
        y = crop_data['label']
        
        # Create crop mapping
        crop_dist = {
            'rice': 0, 'maize': 1, 'chickpea': 2, 'kidneybeans': 3, 'pigeonpeas': 4,
            'mothbeans': 5, 'mungbean': 6, 'blackgram': 7, 'lentil': 8, 'pomegranate': 9,
            'banana': 10, 'mango': 11, 'grapes': 12, 'watermelon': 13, 'muskmelon': 14,
            'apple': 15, 'orange': 16, 'papaya': 17, 'coconut': 18, 'cotton': 19,
            'jute': 20, 'coffee': 21
        }
        
        # Map labels to numbers
        y_numeric = y.map(crop_dist)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)
        
        # Scale features
        crop_scaler = StandardScaler()
        X_train_scaled = crop_scaler.fit_transform(X_train)
        X_test_scaled = crop_scaler.transform(X_test)
        
        # Train model
        crop_model = DecisionTreeClassifier()
        crop_model.fit(X_train_scaled, y_train)
        
        print("Crop model loaded and trained successfully")
        return True
        
    except Exception as e:
        print(f"Error loading crop model: {e}")
        return False

def load_fertilizer_model():
    """Load and train the fertilizer recommendation model"""
    global fertilizer_model, fertilizer_scaler, soil_encoder, crop_encoder
    
    try:
        # Load the dataset
        fertilizer_data = pd.read_csv(FERTILIZER_DATASET_PATH)
        
        # Encode categorical variables
        soil_encoder = LabelEncoder()
        crop_encoder = LabelEncoder()
        
        fertilizer_data['Soil Type'] = soil_encoder.fit_transform(fertilizer_data['Soil Type'])
        fertilizer_data['Crop Type'] = crop_encoder.fit_transform(fertilizer_data['Crop Type'])
        
        # Create fertilizer mapping
        fert_dict = {
            'Urea': 1, 'DAP': 2, '14-35-14': 3, '28-28': 4, 
            '17-17-17': 5, '20-20': 6, '10-26-26': 7
        }
        
        fertilizer_data['fert_no'] = fertilizer_data['Fertilizer Name'].map(fert_dict)
        
        # Prepare features and target
        features = ['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
        X = fertilizer_data[features]
        y = fertilizer_data['fert_no']
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        fertilizer_scaler = StandardScaler()
        X_train_scaled = fertilizer_scaler.fit_transform(X_train)
        X_test_scaled = fertilizer_scaler.transform(X_test)
        
        # Train model
        fertilizer_model = DecisionTreeClassifier()
        fertilizer_model.fit(X_train_scaled, y_train)
        
        print("Fertilizer model loaded and trained successfully")
        return True
        
    except Exception as e:
        print(f"Error loading fertilizer model: {e}")
        return False

@app.route('/api/predict-crop', methods=['POST'])
def predict_crop():
    """Predict crop recommendation based on soil and environmental conditions"""
    try:
        data = request.json
        
        # Validate input data
        required_fields = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prepare input data
        features = np.array([[data['N'], data['P'], data['K'], data['temperature'], 
                            data['humidity'], data['ph'], data['rainfall']]])
        
        # Scale features
        features_scaled = crop_scaler.transform(features)
        
        # Make prediction
        prediction = crop_model.predict(features_scaled)[0]
        
        # Map prediction back to crop name
        crop_dist = {
            0: 'rice', 1: 'maize', 2: 'chickpea', 3: 'kidneybeans', 4: 'pigeonpeas',
            5: 'mothbeans', 6: 'mungbean', 7: 'blackgram', 8: 'lentil', 9: 'pomegranate',
            10: 'banana', 11: 'mango', 12: 'grapes', 13: 'watermelon', 14: 'muskmelon',
            15: 'apple', 16: 'orange', 17: 'papaya', 18: 'coconut', 19: 'cotton',
            20: 'jute', 21: 'coffee'
        }
        
        crop_name = crop_dist[prediction]
        result = f"['{crop_name}'] is a best crop to grow in the farm"
        
        return jsonify({'prediction': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    """Predict fertilizer recommendation based on soil and crop conditions"""
    try:
        data = request.json
        
        # Validate input data
        required_fields = ['temperature', 'humidity', 'moisture', 'soilType', 'cropType', 'nitrogen', 'potassium', 'phosphorous']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Prepare input data
        features = np.array([[data['temperature'], data['humidity'], data['moisture'], 
                            data['soilType'], data['cropType'], data['nitrogen'], 
                            data['potassium'], data['phosphorous']]])
        
        # Scale features
        features_scaled = fertilizer_scaler.transform(features)
        
        # Make prediction
        prediction = fertilizer_model.predict(features_scaled)[0]
        
        # Map prediction back to fertilizer name
        fert_dict = {
            1: 'Urea', 2: 'DAP', 3: '14-35-14', 4: '28-28', 
            5: '17-17-17', 6: '20-20', 7: '10-26-26'
        }
        
        fertilizer_name = fert_dict[prediction]
        result = f"['{fertilizer_name}'] is a best fertilizer for the given conditions"
        
        return jsonify({'prediction': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'API is running'})

if __name__ == '__main__':
    print("Loading ML models...")
    
    # Load models
    crop_loaded = load_crop_model()
    fertilizer_loaded = load_fertilizer_model()
    
    if crop_loaded and fertilizer_loaded:
        print("All models loaded successfully!")
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check the dataset files.")
