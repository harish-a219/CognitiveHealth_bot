# NeuroLens AI Cognitive Health Bot
# This is NOT a medical diagnosis tool. It gives wellness insights only.



import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib as plt


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


print(df.isnull().sum())


