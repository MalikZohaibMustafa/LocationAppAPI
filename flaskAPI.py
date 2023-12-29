from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
EARTH_RADIUS = 6371  # Earth radius in kilometers

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Load the saved model
with open("./mechanics_model.pkl", "rb") as file:
    kdtree, train_data, specialty_mapping = pickle.load(file)

def predict_nearest_mechanics(lat, lon, specialties):
    # Query the KD-tree to find the nearest neighbors (initially fetch more results for filtering)
    distances, indices = kdtree.query([[lat, lon]], k=100)
    
    # Convert distances to kilometers
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

@app.route('/', methods=['GET'])
def welcome():
    return jsonify({"message": "Welcome to the Mechanics Locator API! Use POST Api at route /predict with correct data."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        lat = float(data['Latitude'])
        lon = float(data['Longitude'])
        specialties = data['Specialties']
        result = predict_nearest_mechanics(lat, lon, specialties)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)