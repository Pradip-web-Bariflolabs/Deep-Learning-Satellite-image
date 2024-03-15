import pandas as pd
import ee

ee.Authenticate()
ee.Initialize(project='ee-pradipsutar2018')

# Define the geometry
lat = 19.813744
long = 85.824953
geometry = ee.Geometry.Point([long, lat])
# d_OLI = ee.Image.constant(img.get('EARTH_SUN_DISTANCE'))
# Load the Sentinel 2 image
image = ee.ImageCollection("COPERNICUS/S2_SR") \
            .filterBounds(geometry) \
            .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20)) \
            .first()

# Calculate pH
ph = ee.Image(8.339).subtract(ee.Image(0.827).multiply(image.select('B1').divide(image.select('B8')))).rename('pH')
# Calculate Dissolved Oxygen
dissolved_oxygen = ee.Image(-0.0167).multiply(image.select('B8')) \
                    .add(ee.Image(0.0067).multiply(image.select('B9'))) \
                    .add(ee.Image(0.0083).multiply(image.select('B11'))) \
                    .add(ee.Image(9.577)).rename('Dissolved_Oxygen')

# Calculate NDVI (Normalized Difference Vegetation Index)
ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

# Calculate NDTI (Normalized Difference Turbidity Index)
ndti = image.normalizedDifference(['B4', 'B3']).rename('NDTI')

# Calculate GCI (Green Chlorophyll Index)
gci = image.select('B3').pow(0.5).multiply(image.select('B4')).pow(0.5).subtract(1).rename('GCI')

# Calculate NDCI (Normalized Difference Chlorophyll Index)
ndci = (image.select('B5').subtract(image.select('B4'))).divide(image.select('B5').add(image.select('B4'))).rename('NDCI')

# Calculate NDWI (Normalized Difference Water Index)
ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')

# Calculate Total Suspended Solids (TSS)
tss = image.select('B4').subtract(image.select('B8')).pow(2).multiply(0.6113).rename('TSS')

# Calculate CDOM index
cdom_index = image.select('B3').divide(image.select('B2')).rename('CDOM_Index')

# Reduce the images to get the mean values
ph_mean = ph.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('pH')
dissolved_oxygen_mean = dissolved_oxygen.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('Dissolved_Oxygen')
ndvi_mean = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('NDVI')
ndti_mean = ndti.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('NDTI')
gci_mean = gci.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('GCI')
ndci_mean = ndci.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('NDCI')
ndwi_mean = ndwi.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('NDWI')
tss_mean = tss.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('TSS')
cdom_mean = cdom_index.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=10).get('CDOM_Index')

# Create a dictionary to store the data
data = {
    'pH': ph_mean.getInfo(),
    'Dissolved Oxygen': dissolved_oxygen_mean.getInfo(),
    'NDVI': ndvi_mean.getInfo(),
    'NDTI': ndti_mean.getInfo(),
    'GCI': gci_mean.getInfo(),
    'NDCI': ndci_mean.getInfo(),
    'NDWI': ndwi_mean.getInfo(),
    'TSS': tss_mean.getInfo(),
    'CDOM':cdom_mean.getInfo(),
    'AQUATIC_MACROPYTES':ndvi_mean.getInfo(),
}

# Create a pandas DataFrame
df = pd.DataFrame(data, index=[0])

# Save DataFrame to CSV file
df.to_csv('water_quality_data.csv', index=False)

print("Data saved successfully.")

