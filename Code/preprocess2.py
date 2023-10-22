import glob
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the path to the data and the list of views to be used
file_path = "OCT/data/CroppedYalePNG/"
view_list = ['P00A+025E+00', 'P00A+110E+65', 'P00A-050E+00', 'P00A+035E+15', 'P00A+110E-20', 'P00A-050E-40', 'P00A+035E+40', 
             'P00A+120E+00', 'P00A-060E+20', 'P00A+000E+00', 'P00A+035E+65', 'P00A+130E+20', 'P00A-060E-20', 'P00A+000E+20', 
             'P00A+035E-20']

# Get list of all image files
image_files = glob.glob(file_path + "*.png")

new_size = (168//2, 192//2)

# Select a subset of image files based on the view_list
selected_files = [f for f in image_files if any(view in f for view in view_list)]
print(f"Selected {len(selected_files)} images out of {len(image_files)} total images.")

# Load the images and create the data matrix D
labels = []
D = []

for file in selected_files:
    label = int(file.split("_")[0].split("/")[-1].replace('yaleB',''))
    # Only select images for person 1, 2
    if label in [1, 2, 3, 4, 5, 6, 7, 8]:
        image = Image.open(file)
        image = image.convert('L')
        image = image.resize(new_size)  # Resize image
        im = np.array(image)
        im = im.ravel()  # Flatten the image
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
pca = PCA()
principalComponents = pca.fit_transform(D_demean)

# Create a scree plot
explained_variance = pca.explained_variance_ratio_
plt.plot(np.arange(1, len(explained_variance) + 1), np.cumsum(explained_variance))
plt.title('Scree Plot of PCA')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Choose the number of components to keep
# Here, we choose the number at which 95% of the variance is explained
cum_explained_variance = np.cumsum(explained_variance)
#n_components = np.where(cum_explained_variance >= 0.80)[0][0] + 1
n_components = 10
print(f'Chosen number of components: {n_components}')

E = pca.components_[:n_components]

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