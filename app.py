# Basic modules
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

# Passing the modules
import classes
from classes import ManualMinMaxScaler, DateTransformer, RemoveSkewness, FeatureEngineering, Encode, ScaleFeatures

# Flask app
app = Flask(__name__)

# Prevent the error of "No module named 'classes'" by using the following code
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'ManualAdaBoostClassifier':
            from classes import ManualAdaBoostClassifier
            return ManualAdaBoostClassifier
        elif name == 'DecisionStump':
            from classes import DecisionStump
            return DecisionStump
        return super().find_class(module, name)

# Load the model
with open('data/model.pkl', 'rb') as f:
    model = CustomUnpickler(f).load()

# Main page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])

def predict():
    # Features
    features = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Complain']
    values = []

    # Get the values from the form
    for feature in features:
        if feature in ['Education', 'Marital_Status', 'Dt_Customer']:
            values.append(request.form[feature])

        else:
            values.append(int(request.form[feature]))

    # Transform the data
    data = pd.DataFrame([values], columns=features)
    data = DateTransformer().transform(data)
    data = RemoveSkewness().transform(data)
    data = FeatureEngineering().transform(data)
    data = Encode().transform(data)
    data = ScaleFeatures().transform(data)

    # Prediction
    pred = model.predict(data)
    pred_proba = model.predict_proba(data)

    # Results
    result = 'Accepted' if pred[0] == 1 else 'Rejected'
    percentage = str(round(pred_proba[0][1]*100, 2)) + '%'
    
    # Return to page
    return render_template('index.html', prediction_text='Percentage Customer Accepting: {}'.format(percentage), results = result)  

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
