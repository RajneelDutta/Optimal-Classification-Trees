import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the path to the data
file_path = "OCT/data/CroppedYalePNG/"

# Get list of all image files
image_files = glob.glob(file_path + "*.png")

print(f"Selected {len(image_files)} images.")

# Define the standard size to which all images will be resized
standard_size = (168, 192)

# Load the images and create the data matrix D
labels = []
D = []

for file in image_files:
    image = Image.open(file)
    image = image.convert('L')
    image = image.resize(standard_size)  # Resize image
    im = np.array(image)
    im = im.ravel()  # Flatten the image
    label = int(file.split("_")[0].split("/")[-1].replace('yaleB',''))
    labels.append(label)
    D.append(im)

D = np.array(D)

print("Data matrix created with shape:", D.shape)

# Compute the mean face
average_face = np.mean(D, axis=0)
print("Computed the mean face.")

# Demean the data matrix
D_demean = D - average_face

# Run PCA on the demeaned data matrix
pca = PCA(n_components=10)  # Keep 95% of variance
principalComponents = pca.fit_transform(D_demean)
E = pca.components_

print("PCA applied on the demeaned data matrix. Selected components explaining 80% of variance.")

# Project the data onto the principal components
P = np.dot(D, E.T)

# Split the data into training, validation and test sets
X_temp, X_test, y_temp, y_test = train_test_split(P, labels, test_size=0.2, random_state=1315)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=1315)

# Save the preprocessed data
np.savez('Croppedyalefaces_preprocessed.npz', 
         X_train=X_train, X_val=X_val, X_test=X_test, 
         y_train=y_train, y_val=y_val, y_test=y_test,
         mean_face=average_face, eigenfaces=E)

print("Preprocessed data saved as 'Croppedyalefaces_preprocessed.npz'.")
