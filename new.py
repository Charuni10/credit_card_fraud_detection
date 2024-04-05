from flask import Flask, render_template, request
import datetime
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
with open('rfmodel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create a new StandardScaler instance
scaler = StandardScaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the HTML form
    datetime_value = request.form['datetime']
    gender = int(request.form['gender'])
    category = int(request.form['category'])
    amount = int(request.form['amount'])
    # Parse datetime string into a datetime object
    datetime_obj = datetime.datetime.strptime(datetime_value, '%Y-%m-%dT%H:%M')

    # Extract hour, day, and month
    hour = datetime_obj.hour
    day = datetime_obj.day
    month = datetime_obj.month
    # Fit the scaler to the new data
    scaler.fit([[hour, day, month,category,gender,amount]])

    # Scale the features using the fitted scaler
    scaled_features = scaler.transform([[hour, day, month,category,gender,amount]])

    # Make a prediction
    prediction = model.predict(scaled_features)

    # Return the prediction to the user
    prediction_text = "Not a fraud" if prediction[0] == 0 else "Fraud"

    # Return the prediction to the user
    return render_template('index.html', prediction_text=prediction_text)
if __name__ == '__main__':
    app.run(debug=True)
