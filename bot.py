# NeuroLens AI Cognitive Health Bot
# This is NOT a medical diagnosis tool. It gives wellness insights only.


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

exercise_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df = pd.read_csv('human_cognitive_performance.csv')
# Preprocess the data and cleaning the data

df.drop('Gender', axis=1, inplace=True)
df.drop('Diet_Type', axis=1, inplace=True)
df.drop('Caffeine_Intake', axis=1, inplace=True)

# Handle missing values
df.fillna({
    'Sleep_Duration': df['Sleep_Duration'].mean(),
    'Exercise_Frequency': 'Medium',
    'Stress_Level': df['Stress_Level'].mean(),
    'Cognitive_Score': df['Cognitive_Score'].mean(),
    'Diet_Type': 'Non-Vegetarian'}, inplace=True)
df['Exercise_Frequency'] = df['Exercise_Frequency'].map(exercise_mapping)


#Define features and target variable
X = df[['Age',
        'Sleep_Duration',
        'Exercise_Frequency',
        'Stress_Level',
        'Daily_Screen_Time',
        'Memory_Test_Score',    
        'Reaction_Time']]
y = df['Cognitive_Score']

# Split the data into training and testing sets
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred.round()))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred.round()))

"""FASTAPI implementation for the NeuroLens AI Cognitive Health Bot"""

from fastapi import FastAPI
from pydantic import BaseModel 

app = FastAPI()

# user input structure
class UserInput(BaseModel):
    Age: int
    Sleep_Duration: float
    Exercise_Frequency: str
    Stress_Level: float
    Daily_Screen_Time: float
    Memory_Test_Score: float
    Reaction_Time: float

# prediction route
@app.post("/predict")
def predict(data: UserInput):

    exercise_value = exercise_mapping[data.Exercise_Frequency]

    features = np.array([[
        data.Age,
        data.Sleep_Duration,
        exercise_value,
        data.Stress_Level,
        data.Daily_Screen_Time,
        data.Memory_Test_Score,
        data.Reaction_Time
    ]])

    prediction = rf_model.predict(features)

    return {
        "cognitive_score": round(float(prediction[0]), 2)
    }
