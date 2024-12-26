import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Streamlit App
st.title("Heart Disease Prediction App")

# Preloaded dataset
st.write("### Using Preloaded Dataset")

# Example dataset
data = pd.DataFrame({
    "Age": [63, 37, 41, 56, 57],
    "Female": [0, 1, 0, 0, 1],
    "Male": [1, 0, 1, 1, 0],
    "Cholesterol": [233, 250, 204, 236, 354],
    "Rest BP": [145, 130, 130, 120, 132],
    "Exercise Heart Rate": [150, 187, 172, 178, 163],
    "Old Peak": [2.3, 3.5, 1.4, 0.8, 0.6],
    "FBS": [1, 0, 0, 0, 0],
    "EI Angina": [0, 0, 0, 0, 0],
    "CA": [0, 0, 0, 0, 0],
    "Severity": [0, 1, 0, 1, 1]
})

st.write("Dataset Preview:")
st.write(data.head())

# Prepare data
x = data.drop(columns=["Severity"], axis=1)
y = data["Severity"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection
model_choice = st.selectbox("Select a model", ["KNeighborsClassifier", "RandomForestClassifier", "SVC"])

# Define models and hyperparameters
models = {
    "KNeighborsClassifier": {
        "model": KNeighborsClassifier(),
        "parameters": {
            "n_neighbors": [1, 3, 5],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
        "scale": False,
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(),
        "parameters": {
            "n_estimators": [50, 100, 200],
            "criterion": ["entropy", "gini"],
        },
        "scale": False,
    },
    "SVC": {
        "model": SVC(class_weight="balanced"),
        "parameters": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": [0.01, 0.1, 1],
        },
        "scale": True,
    },
}

selected_model = models[model_choice]

# Use scaled or unscaled data
X_train_to_use = X_train_scaled if selected_model["scale"] else X_train
X_test_to_use = X_test_scaled if selected_model["scale"] else X_test

# Hyperparameter tuning
st.write("Tuning hyperparameters...")
try:
    search = GridSearchCV(estimator=selected_model["model"], param_grid=selected_model["parameters"], cv=5)
    search.fit(X_train_to_use, y_train)

    # Evaluation
    predictions = search.predict(X_test_to_use)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    st.write(f"Best Parameters: {search.best_params_}")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"F1 Score: {f1:.2f}")

    # Real-time prediction form
    st.write("### Predict on a single instance")
    st.write("Fill in the details below:")

    # Input form for user features
    input_data = {
        "Age": st.number_input("Age", min_value=0, max_value=120, value=30),
        "Female": st.selectbox("Female", [0, 1]),
        "Male": st.selectbox("Male", [0, 1]),
        "Cholesterol": st.number_input("Cholesterol", min_value=0.0, value=200.0),
        "Rest BP": st.number_input("Rest BP", min_value=0.0, value=120.0),
        "Exercise Heart Rate": st.number_input("Exercise Heart Rate", min_value=0.0, value=150.0),
        "Old Peak": st.number_input("Old Peak", min_value=0.0, value=1.0),
        "FBS": st.selectbox("Fasting Blood Sugar (FBS)", [0, 1]),
        "EI Angina": st.selectbox("Exercise Induced Angina", [0, 1]),
        "CA": st.number_input("CA (Number of major vessels colored)", min_value=0, max_value=4, value=0),
    }

    # Create DataFrame for input
    input_df = pd.DataFrame([input_data])

    # Scale input if required
    if selected_model["scale"]:
        input_df = scaler.transform(input_df)
    single_prediction = search.predict(input_df)

    # Interpret prediction
    st.write(f"Prediction for the input instance: {single_prediction[0]}")
    severity_map = {
        0: "You may have no heart disease",
        1: "You may have mild heart disease",
        2: "You may have moderate heart disease",
        3: "You may have severe heart disease",
        4: "You may be in advanced stage",
    }
    st.write(severity_map.get(single_prediction[0], "Unknown severity level"))

except Exception as e:
    st.error(f"An error occurred: {e}")
