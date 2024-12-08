from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html',prediction="")




def predict(userInput):
    target_column = 'Sale Amount'
    numerical_features = ['Assessed Value', 'Sales Ratio', 'List Year']
    categorical_features = ['Town', 'Property Type', 'Residential Type']

    # Ensure all preprocessing objects are fitted during training
    # These objects should already be fitted on training data
    global numerical_imputer, scaler, categorical_imputer, encoder

    # Convert user input to DataFrame
    user_df = pd.DataFrame([userInput])

    # Preprocess numerical features
    user_num = pd.DataFrame(numerical_imputer.transform(user_df[numerical_features]), columns=numerical_features)
    user_num_scaled = pd.DataFrame(scaler.transform(user_num), columns=numerical_features)

    # Preprocess categorical features
    user_cat = pd.DataFrame(categorical_imputer.transform(user_df[categorical_features]), columns=categorical_features)
    user_cat_encoded = pd.DataFrame(encoder.transform(user_cat), columns=encoder.get_feature_names_out(categorical_features))

    # Combine preprocessed features
    user_preprocessed = pd.concat([user_num_scaled, user_cat_encoded], axis=1)

    # Make prediction
    pred = regressor.predict(user_preprocessed)

    return int(pred)





@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        # Input
        town = request.form['town']
        property_type = request.form['property_type']
        residential_type = request.form['residential_type']
        Avalue = request.form['Avalue']
        sales_ratio = request.form['sales_ratio']
        List_year = request.form['List_year']

        user_input = {
            "Assessed Value": float(Avalue),
            "Sales Ratio": float(sales_ratio),
            "List Year": int(List_year),
            "Town": town,
            "Property Type": property_type,
            "Residential Type": residential_type
        }

        z_pred = predict(user_input)
    return render_template('home.html', prediction=z_pred)

if __name__ == '__main__':
    df1= pd.read_csv('/Users/akshayamamidi/Downloads/HPP/Real_Estate_Sales_2001-2022_GL.csv',low_memory=False)
    print(df1.head(1))
    target_column = 'Sale Amount'
    numerical_features = ['Assessed Value', 'Sales Ratio', 'List Year']
    categorical_features = ['Town', 'Property Type', 'Residential Type']

    df1 = df1.dropna(subset=[target_column])

    X = df1[numerical_features + categorical_features]
    y = df1[target_column]

    numerical_imputer = SimpleImputer(strategy='mean')
    X_num = pd.DataFrame(numerical_imputer.fit_transform(X[numerical_features]), columns=numerical_features)

    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=numerical_features)

    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X_cat = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_features]), columns=categorical_features)

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_encoded = pd.DataFrame(encoder.fit_transform(X_cat), columns=encoder.get_feature_names_out(categorical_features))
    print("Hiiiiii")
    # Loading saved model
    regressor = joblib.load('model_rf30.sav')
    app.run(debug=True)