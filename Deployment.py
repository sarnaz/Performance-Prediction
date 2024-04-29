from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler



app = Flask(__name__)

# Load trained Random Forest model
model = load('predictionmodel.joblib')
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

def preprocess_data(input_data):
    # Separate the types of data
    categorical_data = input_data[:, :3]  
    numerical_data = input_data[:, 3:]  

    #encode the categorical data
    encoded_data = encoder.transform(categorical_data) 
    
    #standardise the numerical data
    scaled_numerical_data = scaler.transform(numerical_data)

    #join them together
    preprocessed_data = np.concatenate((encoded_data, scaled_numerical_data), axis=1) 
    
    return preprocessed_data  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #Collect data from the form
    form_data = request.form
    features = [
        form_data['preferred_foot'],
        form_data['defensive_work_rate'],
        form_data['attacking_work_rate'],
        form_data['crossing'],
        form_data['finishing'],
        form_data['heading_accuracy'],
        form_data['short_passing'],
        form_data['volleys'],
        form_data['dribbling'],
        form_data['free_kick_accuracy'],
        form_data['long_passing'],
        form_data['ball_control'],
        form_data['acceleration'],
        form_data['sprint_speed'],
        form_data['agility'],
        form_data['reactions'],
        form_data['balance'],
        form_data['shot_power'],
        form_data['jumping'],
        form_data['stamina'],
        form_data['strength'],
        form_data['long_shots'],
        form_data['aggression'],
        form_data['interceptions'],
        form_data['positioning'],
        form_data['vision'],
        form_data['penalties'],
        form_data['marking'],
        form_data['standing_tackle'],
        form_data['sliding_tackle'],
        form_data['gk_diving'],
        form_data['gk_handling'],
        form_data['gk_kicking'],
        form_data['gk_positioning'],
        form_data['gk_reflexes']
    ]
    features = np.array([features]) 
    #features = np.array([form_data])

    #Preprocess this data 
    preprocessed_features = preprocess_data(features)
    
    #Make prediction

    prediction = model.predict(preprocessed_features)
    output = round(prediction[0])
    
    return render_template('index.html', prediction_text='Predicted Overall Rating: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)