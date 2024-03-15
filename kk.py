# # Standard Python libraies
# import os
# import gc

# from IPython.display import display
# from tqdm.notebook import tqdm
# tqdm.pandas()

# # For image processings
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# # Examine
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split 
# # import lightgbm as lgb

# np.random.seed(42)

# Images, Masks = [], []
# # looking throuogh all files under the dataset folder
# for dirname, _, filenames in os.walk("archive\Water Bodies Dataset"):
#     for filename in tqdm(filenames):
#         # open the image using Image library 
#         image = Image.open(os.path.join(dirname, filename))
#         # resize the image, for ease of use
#         image = image.resize((128, 128))
#         # Turn into numpy array
#         image = np.array(image)
        
#         # if the file is from image folder, add the loaded image to Images list
#         if  dirname == "archive\Water Bodies Dataset\Images":
#             Images.append(image)
#         # Otherwise, it is a mask, so add the loaded image to Maskes list
#         else:
#             Masks.append(np.where(image > 128, 255, 0))
#         del image
# # print(Images)

# # clean anything that is Irrelevant from the memory
# gc.collect()

# plt.figure(figsize=(10, 5))
# for i in range(1):
#     plt.subplot(1, 2, 3*i + 1)
#     plt.imshow(Images[i])
#     plt.axis('off')
#     plt.subplot(1, 2, 3*i + 2)
#     plt.imshow(Masks[i])
#     plt.axis('off')

# def to_gradient(array):
#     """
#     change the shape of array from (128, 128) to (128, 128, 3) 
#     so that we can plot the image correctly.
#     """
#     return np.array([[[val, val, val] for val in x]for x in array])
# plt.figure(figsize=(20, 5))

# # red channel
# plt.subplot(1, 4, 1)
# plt.imshow(to_gradient(Images[i][:, :, 0]))
# plt.title("Red Channel")

# # green channel
# plt.subplot(1, 4, 2)
# plt.imshow(to_gradient(Images[i][:, :, 1]))
# plt.title("Green Channel")

# # blue channel
# plt.subplot(1, 4, 3)
# plt.imshow(to_gradient(Images[i][:, :, 2]))
# plt.title("Blue Channel")

# plt.show()

######################################################################################################################################

import os
import numpy as np
from skimage import io, exposure

def normalize_image(image):
    
    image = image.astype(np.float32)
    
    # Normalize pixel values to the range [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    
    return normalized_image

# def preprocess_images(input_dir, output_dir):
   
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # List all files in the input directory
#     image_files = os.listdir(input_dir)
    
#     # Preprocess each image
#     for filename in image_files:
#         # Load image
#         image_path = os.path.join(input_dir, filename)
#         image = io.imread(image_path)
        
#         # Normalize pixel values
#         normalized_image = normalize_image(image)
        
#         # Save preprocessed image
#         output_path = os.path.join(output_dir, filename)
#         io.imsave(output_path, normalized_image)
        
#         print(f"Preprocessed and saved: {output_path}")

# # Example usage
# input_directory = "New folder"
# output_directory = "preprocessed_images"

# preprocess_images(input_directory, output_directory)

######################################################################################################################################

# import os, rgb2gray
# from skimage import exposure,io

# def color_balance_image(image):
#     """
#     Perform color balancing (histogram equalization) on the input image.
    
#     Args:
#     - image: Input image (numpy array).
    
#     Returns:
#     - Color-balanced image (numpy array).
#     """
#     # Convert image to grayscale if it's a multi-channel image
#     if len(image.shape) > 2:
#         image = rgb2gray(image)
    
#     # Perform histogram equalization
#     balanced_image = exposure.equalize_hist(image)
    
#     return balanced_image

#######################################################################################################

# from scipy.ndimage import median_filter

# def reduce_noise(image, kernel_size=3):
#     """
#     Reduce noise in the input image using median filtering.
    
#     Args:
#     - image: Input image (numpy array).
#     - kernel_size: Size of the median filter kernel.
    
#     Returns:
#     - Image with reduced noise (numpy array).
#     """
#     filtered_image = median_filter(image, size=kernel_size)
#     return filtered_image

#######################################################################################################

# from skimage import exposure

# def enhance_image(image):
#     """
#     Enhance the input image using contrast stretching.
    
#     Args:
#     - image: Input image (numpy array).
    
#     Returns:
#     - Enhanced image (numpy array).
#     """
#     # Apply contrast stretching
#     enhanced_image = exposure.rescale_intensity(image)
    
#     return enhanced_image


# def preprocess_images(input_dir, output_dir):
#     """
#     Preprocess all images in the input directory and save them to the output directory.
    
#     Args:
#     - input_dir: Directory containing input images.
#     - output_dir: Directory to save preprocessed images.
#     """
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     # List all files in the input directory
#     image_files = os.listdir(input_dir)
    
#     # Preprocess each image
#     for filename in image_files:
#         # Load image
#         image_path = os.path.join(input_dir, filename)
#         image = io.imread(image_path)
        
#         # Normalize pixel values
#         normalized_image = normalize_image(image)
        
#         # Color balance image
#         balanced_image = enhance_image(normalized_image)
        
#         # Save preprocessed image
#         output_path = os.path.join(output_dir, filename)
#         io.imsave(output_path, balanced_image)
        
#         print(f"Preprocessed and saved: {output_path}")

# input_img = 'preprocessed_image'
# output_img = 'preprocessed_image'

# preprocess_images(input_img,output_img)

# import numpy as np
# import rasterio

# def extract_spectral_features(image_path):
#     """
#     Extract spectral features from a satellite image.
    
#     Args:
#     - image_path: Path to the input satellite image.
    
#     Returns:
#     - Dictionary containing spectral features.
#     """
#     # Open the image file
#     with rasterio.open(image_path) as src:
#         # Read the image as a numpy array
#         image = src.read()

#     # If the image has only one band, convert it to a single-band array
#     if len(image.shape) == 2:
#         image = np.expand_dims(image, axis=0)

#     # Calculate spectral features
#     spectral_features = {
#         'mean_band1': np.mean(image[0]),
#         'std_band1': np.std(image[0]),
#         'min_band1': np.min(image[0]),
#         'max_band1': np.max(image[0]),
#     }

#     return spectral_features

# # Example usage
# image_path = 'preprocessed_image\LC09_L2SP_139046_20240212_20240213_02_T1_SR_B3.TIF'
# spectral_features = extract_spectral_features(image_path)
# print("Spectral Features:", spectral_features)


# import numpy as np

# def min_max_scaling(data):
#     """
#     Perform Min-Max Scaling on the input data.
    
#     Args:
#     - data: Input data (numpy array).
    
#     Returns:
#     - Scaled data (numpy array).
#     """
#     # Calculate the minimum and maximum values for each feature
#     min_vals = np.min(data, axis=0)
#     max_vals = np.max(data, axis=0)
    
#     # Apply Min-Max Scaling to each feature
#     scaled_data = (data - min_vals) / (max_vals - min_vals)
    
#     return scaled_data, min_vals, max_vals

# # Example usage
# # Suppose 'data' is your feature matrix, where each row represents a sample and each column represents a feature
# # For demonstration purposes, let's create a random dataset
# data = np.random.rand(100, 5)  # 100 samples, 5 features

# # Perform Min-Max Scaling
# scaled_data, min_vals, max_vals = min_max_scaling(data)

# # Print scaled data and scaling parameters
# print("Scaled Data:")
# print(scaled_data)
# print("\nMinimum Values:")
# print(min_vals)
# print("\nMaximum Values:")
# print(max_vals)


from sklearn.model_selection import train_test_split

def data_split(X, y, test_size=0.15, validation_size=0.15, random_state=None):
    """
    Split the dataset into training, validation, and testing sets.
    
    Args:
    - X: Input features (numpy array or pandas DataFrame).
    - y: Target variable (numpy array or pandas Series).
    - test_size: Proportion of the dataset to include in the test split.
    - validation_size: Proportion of the dataset to include in the validation split.
    - random_state: Seed for random number generation.
    
    Returns:
    - Tuple containing the training, validation, and testing sets: (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Split the dataset into training and temporary sets
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Split the temporary set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp, test_size=validation_size/(1-test_size), random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Example usage
# X is your feature matrix, y is your target variable
# For demonstration purposes, let's create sample data
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.randint(2, size=100)  # Binary target variable

# Split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = data_split(X, y, test_size=0.15, validation_size=0.15, random_state=42)

# Print the shapes of the resulting sets
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

































# import os
# import numpy as np
# import rasterio

# # Specify the path to the image file
# image_file = "./LC09_L2SP_139046_20240212_20240213_02_T1_SR_B5.TIF"

# # Open the image file using rasterio
# with rasterio.open(image_file, 'r') as img:
#     # Read the image data as a NumPy array
#     image_data = img.read(1)  # Assuming a single-band image, change the band index if necessary

# # Define the output text file path
# output_file = "image_data.txt"

# # Save the NumPy array into a text file
# np.savetxt(output_file, image_data)

# # Print a message to confirm that the file has been saved
# print(f"Image data has been saved to '{output_file}'")
