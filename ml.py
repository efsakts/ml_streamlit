import streamlit as st
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

# Load your dataset
exam_data = pd.read_csv('exams.csv')

# Function to preprocess and encode data


def preprocess_data(data):
    encoding_map = {'gender': {'female': 0, 'male': 1},
                    'race/ethnicity': {'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4},
                    'parental level of education': {'some college': 0, 'high school': 1, "associate's degree": 2,
                                                    'some high school': 3, "bachelor's degree": 4, "master's degree": 5},
                    'lunch': {'standard': 0, 'free/reduced': 1},
                    'test preparation course': {'none': 0, 'completed': 1}}

    data.replace(encoding_map, inplace=True)
    return data


# Preprocess the data
exam_data = preprocess_data(exam_data)

# Define the Streamlit UI
st.title("Math Score Prediction App")

st.header("Input Features")
gender = st.selectbox("Gender", ["female", "male"])
ethnicity = st.selectbox(
    "Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
education = st.selectbox("Parental Education", [
                         "some college", "high school", "associate's degree", "some high school", "bachelor's degree", "master's degree"])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.slider("Reading Score", 0, 100, 50)
writing_score = st.slider("Writing Score", 0, 100, 50)

# Create a button to make predictions
if st.button("Predict Math Score"):
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [ethnicity],
        'parental level of education': [education],
        'lunch': [lunch],
        'test preparation course': [test_prep],
        'reading score': [reading_score],
        'writing score': [writing_score]
    })

    # Preprocess the user input data
    user_input = preprocess_data(user_input)

    # Select relevant features
    features = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course',
                'reading score', 'writing score']
    X = exam_data[features]
    y = exam_data['math score']
    X_user = user_input[features]

    # Train a Decision Tree Classifier
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # Predict the math score
    predicted_math_score = clf.predict(X_user)[0]

    # Display the predicted math score to the user
    st.success(f"Predicted Math Score: {predicted_math_score}")