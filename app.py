import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
from xgboost import XGBRegressor

app = Flask(__name__)

# Load the trained model (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

# Define the order of features used during training
feature_order = ['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from the form
        age = float(request.form['Age'])
        height = float(request.form['Height'])
        weight = float(request.form['Weight'])
        duration = float(request.form['Duration'])
        heart_rate = float(request.form['Heart_Rate'])
        body_temp = float(request.form['Body_Temp'])
        gender = int(request.form['Gender'])

        # Make a prediction
        input_data = [[gender, age, height, weight, duration, heart_rate, body_temp]]
        prediction = model.predict(pd.DataFrame(input_data, columns=feature_order))

        # Round the prediction to two decimal places
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Predicted Calories: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(port=5001)
