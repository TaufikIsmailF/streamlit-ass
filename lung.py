import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Title of the App
st.title("Lung Cancer Analysis Web App")

# 1. Caching the data loading function
@st.cache_data
def load_data():
    # Load your dataset (adjust the path accordingly)
    df = pd.read_csv('/Users/taufikismail/Documents/GitHub/streamlit-ass/lungcancer.csv') 
    return df

# Load the dataset
df = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Select a page:", ['Home', 'Data Visualization', 'Prediction'])

# 2. Visualization Page
if option == 'Data Visualization':
    st.header("Data Visualizations")

    # Visualization 1: Gender Distribution
    st.subheader("1. Gender Distribution in Lung Cancer Dataset")
    fig1 = px.histogram(df, x="GENDER", title="GENDER Distribution", color="LUNG_CANCER", barmode='group')
    st.plotly_chart(fig1)

    # Visualization 2: Age Distribution with Tabs
    st.subheader("2. Age Distribution by Cancer Status")
    tab1, tab2 = st.tabs(["All AGES", "By GENDER"])
    
    with tab1:
        fig2 = px.histogram(df, x="AGE", title="Age Distribution (All)")
        st.plotly_chart(fig2)
    
    with tab2:
        selected_gender = st.selectbox("Select Gender", df["GENDER"].unique())
        filtered_df = df[df["GENDER"] == selected_gender]
        fig3 = px.histogram(filtered_df, x="AGE", title=f"Age Distribution for {selected_gender}")
        st.plotly_chart(fig3)

    
# Load your DataFrame here (df)
# Assuming df is already defined somewhere above

# Convert categorical variables to numerical values
df['GENDER_NUM'] = df['GENDER'].map({'M': 1, 'F': 0})
df['SMOKING_NUM'] = df['SMOKING'].map({'YES': 1, 'NO': 0})

# Set features and target variable
X = df[['AGE', 'GENDER_NUM', 'SMOKING_NUM']]
y = df['LUNG_CANCER']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction section
if option == 'Prediction':
    st.header("Lung Cancer Prediction using Random Forest")

    # User inputs for prediction
    st.subheader("Input features for prediction")
    age = st.slider("AGE", int(df["AGE"].min()), int(df["AGE"].max()), 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking Habit", ["Yes", "No"])

    # Convert user input to numerical values
    gender_num = 1 if gender == "Male" else 0
    smoking_num = 1 if smoking == "Yes" else 0

    # Make a prediction
    prediction = model.predict([[age, gender_num, smoking_num]])

    # Display the prediction result
    if prediction[0] == 1:
        st.write("### Prediction: High risk of Lung Cancer")
    else:
        st.write("### Prediction: Low risk of Lung Cancer")

    # Evaluate the model
    st.write("### Model Accuracy")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy * 100:.2f}%")