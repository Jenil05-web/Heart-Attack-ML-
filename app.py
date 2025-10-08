from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for making a prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    # The order of features MUST match the order they were trained on
    features = [float(x) for x in request.form.values()]
    
    # Arrange the features into a NumPy array
    final_features = [np.array(features)]
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(final_features)
    
    # Make the prediction using the loaded model
    prediction = model.predict(scaled_features)
    
    # Determine the output message
    if prediction[0] == 1:
        output = 'High Risk of Heart Attack'
    else:
        output = 'Low Risk of Heart Attack'
        
    # Render the result page with the prediction
    return render_template('result.html', prediction_text=f'Prediction: {output}')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)