import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
import pickle


# Constants
EARTH_RADIUS = 6371  # Earth radius in kilometers

# Data Loading
data_path = "./updated_mechanics_data_google.csv"  
mechanics_data = pd.read_csv(data_path)

# Handling missing values
mechanics_data = mechanics_data.dropna(subset=['Latitude', 'Longitude', 'Specialty'])

# Remove outliers using IQR method for Latitude and Longitude as given in the original code
Q1_lat = mechanics_data['Latitude'].quantile(0.25)
Q3_lat = mechanics_data['Latitude'].quantile(0.75)
IQR_lat = Q3_lat - Q1_lat
lower_bound_lat = Q1_lat - 1.5 * IQR_lat
upper_bound_lat = Q3_lat + 1.5 * IQR_lat
mechanics_data = mechanics_data[(mechanics_data['Latitude'] >= lower_bound_lat) & (mechanics_data['Latitude'] <= upper_bound_lat)]

Q1_lon = mechanics_data['Longitude'].quantile(0.25)
Q3_lon = mechanics_data['Longitude'].quantile(0.75)
IQR_lon = Q3_lon - Q1_lon
lower_bound_lon = Q1_lon - 1.5 * IQR_lon
upper_bound_lon = Q3_lon + 1.5 * IQR_lon
mechanics_data = mechanics_data[(mechanics_data['Longitude'] >= lower_bound_lon) & (mechanics_data['Longitude'] <= upper_bound_lon)]

# # Model Generation (KD-tree)
# specialty_mapping = {specialty: i for i, specialty in enumerate(mechanics_data['Specialty'].unique())}
# mechanics_data['Specialty_encoded'] = mechanics_data['Specialty'].map(specialty_mapping)
# train_data, _ = train_test_split(mechanics_data, test_size=0.2, random_state=42)
# coordinates = train_data[['Latitude', 'Longitude']].values
# kdtree = KDTree(coordinates)

# Binary Encoding for Specialties
unique_specialties = mechanics_data['Specialty'].unique()
for specialty in unique_specialties:
    mechanics_data[specialty] = (mechanics_data['Specialty'] == specialty).astype(int)

# Model Generation (KD-tree)
train_data, _ = train_test_split(mechanics_data, test_size=0.2, random_state=42)
coordinates = train_data[['Latitude', 'Longitude']].values
kdtree = KDTree(coordinates)

# Save the KD-tree and additional data as a .pkl file
with open("./mechanics_model.pkl", "wb") as file:
    pickle.dump((kdtree, train_data, unique_specialties), file)

print("Model saved successfully!")
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
    return nearest_mechanics[['Name', 'ID Number', 'Phone Number', 'Address', 'Specialty', 'Latitude', 'Longitude']].head(5).to_dict(orient="records")

# Predicting for dummy test cases
result = predict_nearest_mechanics('24.8787702', '66.87899999999999', ['Denter', 'Painter'])

print(result)
