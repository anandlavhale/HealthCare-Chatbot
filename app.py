from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load dataset
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns[:-1]  # Features (symptoms)
x = training[cols]
y = training['prognosis']

# Encode disease labels
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(x, y)

# Load severity, description, and precautions
severityDictionary = {}
description_list = {}
precautionDictionary = {}

def load_data():
    global severityDictionary, description_list, precautionDictionary
    severity_data = pd.read_csv('MasterData/symptom_severity.csv')
    for _, row in severity_data.iterrows():
        severityDictionary[row[0]] = int(row[1])

    desc_data = pd.read_csv('MasterData/symptom_Description.csv')
    for _, row in desc_data.iterrows():
        description_list[row[0]] = row[1]

    precaution_data = pd.read_csv('MasterData/symptom_precaution.csv')
    for _, row in precaution_data.iterrows():
        precautionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

load_data()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        symptoms = data.get("symptoms", [])

        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Convert symptoms into model input
        input_features = [1 if feature in symptoms else 0 for feature in cols]
        prediction = clf.predict([input_features])[0]
        disease = le.inverse_transform([prediction])[0]

        # Get description and precautions
        description = description_list.get(disease, "No description available.")
        precautions = precautionDictionary.get(disease, ["No precautions available."])

        response = {
            "disease": disease,
            "description": description,
            "precautions": precautions
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
