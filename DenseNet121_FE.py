import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121  # Updated to DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from imblearn.over_sampling import SMOTE  # Import SMOTE

# Load DenseNet121 model (excluding the top classification layer)
base_model = DenseNet121(weights='imagenet', include_top=False)  # Updated to DenseNet121

# Add Global Average Pooling layer to get a feature vector
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Create the model with the modified architecture
feature_extractor = Model(inputs=base_model.input, outputs=x)

directory = 'D:/......'
if not os.path.exists(directory):
    os.makedirs(directory)


# Get a list of class folders
class_folders = os.listdir(directory)

# Initialize lists to store features and class labels
features_list = []
class_labels = []

# Loop through each class folder
for class_folder in class_folders:
    class_path = os.path.join(directory, class_folder)
    
    # Get a list of image file names in the class folder
    image_file_names = os.listdir(class_path)
    
    # Loop through images within the class folder
    for file_name in image_file_names:
        image_path = os.path.join(class_path, file_name)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        preprocessed_image = tf.keras.applications.densenet.preprocess_input(image_array)  # Updated for DenseNet121
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add an extra dimension
        features = feature_extractor.predict(preprocessed_image)
        features_list.append(features.flatten())  # Flatten the features to avoid dimensionality issues
        class_labels.append(class_folder)  # Store the class label for this image

# Stack features into an array
features_array = np.vstack(features_list)

# Convert class labels into numerical format
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
class_labels_encoded = le.fit_transform(class_labels)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(features_array, class_labels_encoded)

# Convert y_resampled back to original class names if necessary
y_resampled_labels = le.inverse_transform(y_resampled)

# To save the balanced features and labels to CSV (optional)
df_resampled = pd.DataFrame(X_resampled)
df_resampled['class_label'] = y_resampled_labels

# Reorder columns to have class_label as the first column
cols_resampled = df_resampled.columns.tolist()
cols_resampled = cols_resampled[-1:] + cols_resampled[:-1]
df_resampled = df_resampled[cols_resampled]

csv_filename_resampled = os.path.join(directory, '.........csv')
df_resampled.to_csv(csv_filename_resampled, index=False)

print(f"Balanced features and class labels saved to {csv_filename_resampled}")
