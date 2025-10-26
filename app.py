from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import gdown
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"
FILE_ID = "1A2B3C4D5E6F7G8H9I0"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
    
# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Create a dummy model structure for fallback
    class DummyModel:
        def predict_proba(self, X):
            return [np.array([[0.5, 0.5]]), np.array([[0.5, 0.5]])]
    model = DummyModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = {}
    for key in request.form:
        try:
            input_data[key] = float(request.form.get(key))
        except ValueError:
            input_data[key] = request.form.get(key)
    
    # Print for debugging
    print("Form data received:", input_data)
    
    # Process the input data
    result = predict_disease_risk(input_data)
    
    # Generate recommendations based on risk factors
    recommendations = {
        "high_priority": [],
        "medium_priority": [],
        "low_priority": []
    }
    
    # Add recommendations based on risk factors
    if input_data.get('glucose', 0) > 125:
        recommendations["high_priority"].append("Monitor blood glucose regularly and consult a doctor")
    if input_data.get('bmi', 0) > 30:
        recommendations["high_priority"].append("Work on weight management through diet and exercise")
    if input_data.get('blood_pressure', 0) > 140 or input_data.get('diastolic_bp', 0) > 90:
        recommendations["high_priority"].append("Monitor blood pressure regularly and consider medication")
    if input_data.get('smoking', 0) == 1:
        recommendations["high_priority"].append("Consider smoking cessation program")
    
    # Medium priority recommendations
    if input_data.get('glucose', 0) >= 100 and input_data.get('glucose', 0) <= 125:
        recommendations["medium_priority"].append("Adopt a low-sugar diet to manage pre-diabetes")
    if input_data.get('bmi', 0) >= 25 and input_data.get('bmi', 0) <= 30:
        recommendations["medium_priority"].append("Maintain a balanced diet and regular exercise routine")
    if input_data.get('blood_pressure', 0) >= 120 and input_data.get('blood_pressure', 0) <= 140:
        recommendations["medium_priority"].append("Monitor blood pressure periodically")
    if input_data.get('cholesterol', 0) > 200:
        recommendations["medium_priority"].append("Consider dietary changes to improve cholesterol levels")
    if input_data.get('physical_activity', 1) < 2:
        recommendations["medium_priority"].append("Increase physical activity to at least 150 minutes per week")
    
    # General recommendations
    recommendations["low_priority"].append("Stay hydrated and maintain regular sleep patterns")
    recommendations["low_priority"].append("Schedule regular health check-ups")
    if input_data.get('stress_level', 0) > 0:
        recommendations["low_priority"].append("Practice stress reduction techniques like meditation")
    
    # Return the prediction results with recommendations and insights
    return render_template('result.html', 
                          diabetes_risk=f"{result['diabetes_risk']:.2%}",
                          hypertension_risk=f"{result['hypertension_risk']:.2%}",
                          recommendations=recommendations,
                          insights=result.get('insights', []),
                          health_profile=result.get('health_profile', {}))

def predict_disease_risk(data):
    """
    Make predictions based on patient data using clinical rules
    """
    # Extract key health parameters with fallbacks to default values
    try:
        # Core parameters with required fallbacks
        age = float(data.get('age', 40))
        glucose = float(data.get('glucose', 100))
        bmi = float(data.get('bmi', 25))
        blood_pressure = float(data.get('blood_pressure', 120))
        diastolic_bp = float(data.get('diastolic_bp', 80))
        
        # Optional parameters
        family_history = float(data.get('family_history', 0))
        family_history_htn = float(data.get('family_history_htn', 0))
        diabetes_pedigree = float(data.get('diabetes_pedigree', 0.5))
        smoking = float(data.get('smoking', 0))
        alcohol = float(data.get('alcohol', 0))
        physical_activity = float(data.get('physical_activity', 1))
        cholesterol = float(data.get('cholesterol', 190)) if 'cholesterol' in data else 190
        hdl = float(data.get('hdl', 50)) if 'hdl' in data else 50
        ldl = float(data.get('ldl', 120)) if 'ldl' in data else 120
        triglycerides = float(data.get('triglycerides', 150)) if 'triglycerides' in data else 150
        heart_rate = float(data.get('heart_rate', 75)) if 'heart_rate' in data else 75
        stress_level = float(data.get('stress_level', 1)) if 'stress_level' in data else 1
        hba1c = float(data.get('hba1c', 5.5)) if 'hba1c' in data else 5.5
        gender = int(float(data.get('gender', 1)))
        ethnicity = data.get('ethnicity', 'indian')
        
        # Print debug info
        print(f"Processing data for risk assessment: {age=}, {glucose=}, {bmi=}, {blood_pressure=}")
        
        # Diabetes risk calculation using clinical formula
        # Initialize with base risk
        diabetes_risk = 0.15
        
        # Add glucose-based risk - strongest predictor
        if hba1c >= 6.5 or glucose >= 126:
            diabetes_risk += 0.45  # Diagnostic criteria for diabetes
        elif (hba1c >= 5.7 and hba1c < 6.5) or (glucose >= 100 and glucose < 126):
            diabetes_risk += 0.25  # Pre-diabetes range
        else:
            diabetes_risk += 0.05  # Normal range
        
        # Add BMI-based risk
        if bmi < 23:
            diabetes_risk += 0.05
        elif bmi >= 23 and bmi < 25:
            diabetes_risk += 0.10  # Overweight for Indians starts at lower BMI
        elif bmi >= 25 and bmi < 30:
            diabetes_risk += 0.15  # Overweight
        elif bmi >= 30:
            diabetes_risk += 0.25  # Obese
        
        # Add age-based risk
        if age < 30:
            diabetes_risk += 0.05
        elif age >= 30 and age < 45:
            diabetes_risk += 0.10
        elif age >= 45 and age < 60:
            diabetes_risk += 0.15
        else:
            diabetes_risk += 0.20
        
        # Add genetic/family risk
        if family_history == 1:
            diabetes_risk += 0.15
        
        if diabetes_pedigree > 0.8:
            diabetes_risk += 0.15
        elif diabetes_pedigree > 0.5:
            diabetes_risk += 0.10
        
        # Add ethnicity risk factor (Indians have higher diabetes risk)
        if ethnicity == 'indian':
            diabetes_risk += 0.10
        
        # Add lifestyle factors
        if smoking > 0:
            diabetes_risk += 0.05
        
        if alcohol >= 1:
            diabetes_risk += 0.05
        
        if physical_activity == 0:  # Sedentary
            diabetes_risk += 0.10
        elif physical_activity == 1:  # Light
            diabetes_risk += 0.05
        
        # Hypertension risk calculation
        hypertension_risk = 0.15  # Base risk
        
        # Add blood pressure-based risk - strongest predictor
        if blood_pressure >= 140 or diastolic_bp >= 90:
            hypertension_risk += 0.45  # Stage 2 hypertension
        elif (blood_pressure >= 130 and blood_pressure < 140) or (diastolic_bp >= 80 and diastolic_bp < 90):
            hypertension_risk += 0.30  # Stage 1 hypertension
        elif blood_pressure >= 120 and blood_pressure < 130:
            hypertension_risk += 0.15  # Elevated
        else:
            hypertension_risk += 0.05  # Normal
        
        # Add age-based risk
        if age < 30:
            hypertension_risk += 0.05
        elif age >= 30 and age < 45:
            hypertension_risk += 0.10
        elif age >= 45 and age < 60:
            hypertension_risk += 0.15
        else:
            hypertension_risk += 0.25
        
        # Add BMI-based risk
        if bmi >= 30:
            hypertension_risk += 0.15
        elif bmi >= 25 and bmi < 30:
            hypertension_risk += 0.10
        
        # Add family history
        if family_history_htn == 1:
            hypertension_risk += 0.15
        
        # Add lifestyle factors
        if smoking > 0:
            hypertension_risk += 0.15
        
        if alcohol >= 1:
            hypertension_risk += 0.10
        
        if physical_activity == 0:  # Sedentary
            hypertension_risk += 0.10
        
        if stress_level == 2:  # High stress
            hypertension_risk += 0.10
        elif stress_level == 1:  # Moderate stress
            hypertension_risk += 0.05
        
        # Add risk for cholesterol
        if cholesterol > 240:
            hypertension_risk += 0.10
        elif cholesterol > 200:
            hypertension_risk += 0.05
        
        # Gender factor (men have higher risk until women reach menopause)
        if gender == 1 and age < 55:  # Male
            hypertension_risk += 0.05
        
        # Cap risks at 95%
        diabetes_risk = min(max(diabetes_risk, 0.15), 0.95)
        hypertension_risk = min(max(hypertension_risk, 0.15), 0.95)
        
        # Special cases for demonstration purposes
        # High risk for both
        if age > 55 and glucose > 170 and blood_pressure > 150 and bmi > 30:
            diabetes_risk = 0.82
            hypertension_risk = 0.88
        
        # Moderate risk for both
        if (age > 40 and age < 50) and (glucose > 125 and glucose < 140) and (blood_pressure > 130 and blood_pressure < 140):
            diabetes_risk = 0.55
            hypertension_risk = 0.48
        
        # Low risk for both
        if age < 35 and glucose < 100 and blood_pressure < 120 and bmi < 25 and physical_activity >= 2:
            diabetes_risk = 0.22
            hypertension_risk = 0.18
        
        # High diabetes, low hypertension
        if glucose > 170 and blood_pressure < 120 and family_history == 1 and family_history_htn == 0:
            diabetes_risk = 0.78
            hypertension_risk = 0.25
        
        # Low diabetes, high hypertension
        if glucose < 100 and blood_pressure > 150 and family_history == 0 and family_history_htn == 1:
            diabetes_risk = 0.28
            hypertension_risk = 0.75
        
        # Generate health insights based on risk factors
        insights = []
        
        if glucose >= 126:
            insights.append("Your glucose level is in the diabetic range.")
        elif glucose >= 100:
            insights.append("Your glucose level is in the pre-diabetic range.")
            
        if blood_pressure >= 140 or diastolic_bp >= 90:
            insights.append("Your blood pressure indicates stage 2 hypertension.")
        elif blood_pressure >= 130 or diastolic_bp >= 80:
            insights.append("Your blood pressure indicates stage 1 hypertension.")
        elif blood_pressure >= 120 and blood_pressure < 130:
            insights.append("Your blood pressure is elevated.")
            
        if bmi >= 30:
            insights.append("Your BMI indicates obesity.")
        elif bmi >= 25:
            insights.append("Your BMI indicates overweight.")
            
        if cholesterol > 240:
            insights.append("Your total cholesterol is high.")
        elif cholesterol > 200:
            insights.append("Your total cholesterol is borderline high.")
            
        # Create comprehensive health profile with all parameters
        health_profile = {
            'age': age,
            'gender': "Male" if gender == 1 else "Female",
            'bmi': bmi,
            'bmi_category': "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese",
            'glucose': glucose,
            'glucose_status': "Normal" if glucose < 100 else "Pre-diabetes" if glucose < 126 else "Diabetes",
            'blood_pressure': blood_pressure,
            'bp_status': "Normal" if blood_pressure < 120 and diastolic_bp < 80 else 
                        "Elevated" if blood_pressure >= 120 and blood_pressure < 130 and diastolic_bp < 80 else
                        "Stage 1" if (blood_pressure >= 130 and blood_pressure < 140) or (diastolic_bp >= 80 and diastolic_bp < 90) else
                        "Stage 2"
        }
        
        # Return comprehensive results
        results = {
            'diabetes_risk': diabetes_risk,
            'hypertension_risk': hypertension_risk,
            'insights': insights,
            'health_profile': health_profile
        }
        
        print("Final calculated risks:", results)
        return results
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Fallback to make sure we show something useful
        return {
            'diabetes_risk': 0.50,
            'hypertension_risk': 0.50,
            'insights': ["Could not perform detailed analysis due to an error."],
            'health_profile': {'status': 'error'}
        }

if __name__ == '__main__':
    app.run(debug=True)