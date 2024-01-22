from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from haversine import haversine
EARTH_RADIUS = 6371  # Earth radius in kilometers

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Load the saved model
with open("./mechanics_model.pkl", "rb") as file:
    kdtree, train_data, specialty_mapping = pickle.load(file)


@app.route('/', methods=['GET'])
def welcome():
    return jsonify({"message": "Welcome to the Mechanics Locator API! Use POST Api at route /predict with correct data."})


@app.route('/predict', methods=['GET'])
def welcomepredict():
    return jsonify({"message": "This is Browser Based GET request. Use POST request with correct Data to predict nearest machenics."})


def train_random_forest(mechanics_data):
    # Assuming 'ShopDetails' can be used as a proxy for 'Specialty'
    mechanics_data['Specialty'] = mechanics_data['ShopDetails']

    # Convert 'Specialty' to categorical type if it's not already
    mechanics_data['Specialty'] = mechanics_data['Specialty'].astype('category')

    # Convert categorical data to numerical using one-hot encoding
    mechanics_data_encoded = pd.get_dummies(mechanics_data, columns=['Specialty'])

    # Assuming we use a placeholder target variable for the sake of demonstration
    mechanics_data_encoded['Target_Variable'] = np.random.rand(mechanics_data_encoded.shape[0])  # Placeholder

    feature_columns = [col for col in mechanics_data_encoded.columns if col not in ['Target_Variable', 'Latitude', 'Longitude', 'CNIC', 'Name', 'Email', 'PhoneNO', 'Identity', 'Status', 'ShopDetails', 'Phone1', 'Phone2', 'Address']]

    X = mechanics_data_encoded[feature_columns]
    y = mechanics_data_encoded['Target_Variable']  # Placeholder target variable

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    # Return the model and the columns used for training
    return rf_model, X.columns


def predict_nearest_mechanics(lat, lon, specialties):
    distances, indices = kdtree.query([[lat, lon]], k=100)
    
    distances_in_km = distances * EARTH_RADIUS
    
    # Filter the results based on the provided specialties
    nearest_mechanics = train_data.iloc[indices[0]]
    
    # Here, we filter mechanics who match the provided specialties
    nearest_mechanics = nearest_mechanics[nearest_mechanics['Specialty'].isin(specialties)]
    
    # Distance filtering
    distance_mask = distances_in_km[0][:len(nearest_mechanics)] <= 2
    nearest_mechanics = nearest_mechanics[distance_mask]
    
    # Return the details of the top 10 nearest mechanics
    return nearest_mechanics[['Name', 'ID Number', 'Phone Number', 'Address', 'Specialty', 'Latitude', 'Longitude']].head(10).to_dict(orient="records")




def find_nearest_mechanics(user_location, mechanics_data, specialties, num_mechanics=3):
    # Convert specialties to lowercase for case-insensitive comparison
    specialties_lower = [s.lower() for s in specialties]
    mechanics_data['ShopDetails_lower'] = mechanics_data['ShopDetails'].str.lower()

    # Filter by status and specialty
    active_mechanics = mechanics_data[mechanics_data['Status'] == 'Enabled']
    matching_mechanics = active_mechanics[active_mechanics['ShopDetails_lower'].isin(specialties_lower)]

    # Calculate distances
    matching_mechanics['Distance'] = matching_mechanics.apply(lambda row: round(haversine(user_location, (row['Latitude'], row['Longitude'])), 3), axis=1)
    
    if matching_mechanics.empty:
        # If no exact matches, find nearest mechanics with similar specialties
        active_mechanics['SpecialtyMatch'] = active_mechanics['ShopDetails_lower'].apply(lambda x: any(s in x for s in specialties_lower))
        nearest_similar = active_mechanics[active_mechanics['SpecialtyMatch']].sort_values(by='Distance').head(num_mechanics)
        return nearest_similar

    return matching_mechanics.sort_values(by='Distance').head(num_mechanics)




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from request
        data = request.get_json(force=True)
        user_location = (data['Latitude'], data['Longitude'])
        # user_lat = data['Latitude']
        # user_lon = data['Longitude']
        specialties = data['Specialties']
        mechanics_data = pd.DataFrame(data['MechanicsData'])

        # Convert new mechanics data to DataFrame
        mechanics_df = pd.DataFrame(mechanics_data)

        # Adding a placeholder for 'ShopDetails' as 'Specialty'
        mechanics_df['Specialty'] = mechanics_df['ShopDetails']
        #  Train or load your Random Forest model
        rf_model, training_columns = train_random_forest(mechanics_df)
         # Ensure the DataFrame for prediction has the same features as the training DataFrame
        for col in training_columns:
            if col not in mechanics_df:
                mechanics_df[col] = 0

        # Predict specialty scores using Random Forest
        mechanics_df['Specialty_Score'] = rf_model.predict(mechanics_df[training_columns])


        # Ensure all necessary fields are present, fill missing data
        required_fields = ['Latitude', 'Longitude', 'ShopDetails', 'Status']
        for field in required_fields:
            if field not in mechanics_data:
                mechanics_data[field] = np.nan

         # Fill missing data based on nearby data or default values
        mechanics_data['Latitude'].fillna(method='ffill', inplace=True)
        mechanics_data['Longitude'].fillna(method='ffill', inplace=True)
        mechanics_data['ShopDetails'].fillna('General', inplace=True)  # Default to 'General' if specialty is missing

        # Find nearest mechanics
        nearest_mechanics = find_nearest_mechanics(user_location, mechanics_data, specialties)

        # Prepare response data
        response_data = nearest_mechanics[['CNIC', 'Name', 'Latitude', 'Longitude', 'Email', 'ShopDetails', 'Distance']]

        response_data['Phone'] = nearest_mechanics[['Phone2', 'PhoneNO']].apply(lambda x: x.dropna().tolist(), axis=1)

        return jsonify(response_data.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


