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

# File uploader
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file:
    try:
        # Read the CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())

        # Check for the target column
        if "Severity" not in data.columns:
            st.error("The dataset must contain a 'Severity' column.")
        else:
            # Prepare data
            X = data.drop(columns=["Severity"], axis=1)
            y = data["Severity"]

            # Check for missing values
            if X.isnull().sum().any() or y.isnull().any():
                st.error("The dataset contains missing values. Please clean your data before uploading.")
            else:
                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
                input_data = {}
                for col in X.columns:
                    if X[col].dtype in ["int64", "float64"]:
                        input_data[col] = st.number_input(col, value=float(X[col].mean()))
                    else:
                        input_data[col] = st.selectbox(col, sorted(X[col].unique()))

                # Create DataFrame for input
                input_df = pd.DataFrame([input_data])

                # Scale input if required
                single_instance_scaled = scaler.transform(input_df) if selected_model["scale"] else input_df
                single_prediction = search.predict(single_instance_scaled)

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
else:
    st.info("Please upload a dataset to get started.")
