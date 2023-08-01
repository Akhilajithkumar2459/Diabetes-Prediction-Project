import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Pregnancies = st.sidebar.number_input("number of pregnancies")
    Glucose = st.sidebar.number_input("Glucose level")
    BloodPressure = st.sidebar.number_input("BloodPressure level")
    SkinThickness = st.sidebar.number_input("SkinThickness level")
    Insulin = st.sidebar.number_input("Insulin level")
    BMI=st.sidebar.number_input("BMI level")
    DiabetesPedigreeFunction=st.sidebar.number_input("DiabetesPedigreeFunction")
    Age=st.sidebar.number_input("Age")
    data = {'Pregnancies':Pregnancies,
            'Glucose':Glucose,
            'BloodPressure':BloodPressure,
            'SkinThickness':SkinThickness,
            'Insulin':Insulin,
            'BMI':BMI,
            'DiabetesPedigreeFunction':DiabetesPedigreeFunction,
            'Age':Age
            }
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('grid1.pickle', 'rb'))

prediction = loaded_model.predict(df)
Outcome = loaded_model.predict_outcome(df)

st.subheader('Predicted Result')
st.write('Yes' if Outcome == 1 else 'No')
st.subheader('Outcome')
st.write(Outcome)