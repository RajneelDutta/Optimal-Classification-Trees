import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Model
#from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.decomposition import PCA
from keras.layers import GlobalAveragePooling2D

file_path = "/content/drive/MyDrive/OCT/data/CroppedYalePNG/"
view_list = ['P00A+025E+00', 'P00A+110E+65', 'P00A-050E+00', 'P00A+035E+15', 'P00A+110E-20', 'P00A-050E-40', 'P00A+035E+40', 
             'P00A+120E+00', 'P00A-060E+20', 'P00A+000E+00', 'P00A+035E+65', 'P00A+130E+20', 'P00A-060E-20', 'P00A+000E+20', 
             'P00A+035E-20']

image_files = glob.glob(file_path + "*.png")

new_size = (224, 224)  # VGG16 expects input images of size 224x224

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
  layer.trainable = False

#gap = GlobalAveragePooling2D()(base_model.output)

# Define a new model that takes the output of the Global Average Pooling as its output
model = Model(inputs=base_model.input, outputs=base_model.output)

selected_files = [f for f in image_files if any(view in f for view in view_list)]
print(f"Selected {len(selected_files)} images out of {len(image_files)} total images.")

labels = []
images = []

for file in selected_files:
    label = int(file.split("_")[0].split("/")[-1].replace('yaleB',''))
    # Only select images for person 1, 2, 3, 4, 5, 6, 7, 8
    if label in [1, 2]:
        image = Image.open(file)
        image = image.convert('RGB')  # Convert grayscale to RGB
        image = image.resize(new_size)  # Resize image
        im = img_to_array(image)  # Convert image to numpy array
        im = np.expand_dims(im, axis=0)  # Add batch dimension
        im = im / 255.0  # Normalize to [0,1]
        images.append(im[0])
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training, validation and test sets
X_temp, X_test, y_temp, y_test = train_test_split(images, labels, test_size=0.2, random_state=1315)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=1315)

# Extract features using VGG16 for train, validation and test sets
D_train = model.predict(X_train)
D_val = model.predict(X_val)
D_test = model.predict(X_test)

# Flatten the features
D_train = D_train.reshape(D_train.shape[0], -1)
D_val = D_val.reshape(D_val.shape[0], -1)
D_test = D_test.reshape(D_test.shape[0], -1)

print("Data matrix created with shapes:", D_train.shape, D_val.shape, D_test.shape)

# Save the preprocessed data
np.savez('Croppedyalefaces_preprocessed.npz', 
         X_train=D_train, X_val=D_val, X_test=D_test, 
         y_train=y_train, y_val=y_val, y_test=y_test)

print("Preprocessed data saved as 'Croppedyalefaces_preprocessed.npz'.")
