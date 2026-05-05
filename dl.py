# Import Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
# Load and preprocess the image
image_path = r"C:\Users\Varsh\Desktop\cat.webp" # Replace with your image path
image = load_img(image_path, target_size=(224, 224)) # Resize the image
image_array = img_to_array(image)
image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
# Create an ImageDataGenerator object with augmentation settings
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

# Create an iterator
augmented_images = datagen.flow(image_array, batch_size=1)
# Display some augmented images
plt.figure(figsize=(10, 10))
for i in range(9):
    batch = next(augmented_images)
    img = batch[0].astype('uint8')
plt.subplot(3, 3, i + 1)
plt.imshow(img / 255.0)
plt.axis('off')
plt.tight_layout()
plt.show
