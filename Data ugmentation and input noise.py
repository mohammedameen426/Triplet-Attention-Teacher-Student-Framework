import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your input and output directories
input_images_dir = '/content/Deepcrack_2%/train/image/'
input_masks_dir = '/content/Deepcrack_2%/train/label/'
output_dir_img = '/content/train/image/'
output_dir_lab = '/content/train/label/'

# Create the output directory if it doesn't exist
os.makedirs(output_dir_img, exist_ok=True)
os.makedirs(output_dir_lab, exist_ok=True)

# Define the augmentation parameters
rotation_range = 90
horizontal_flip = True
vertical_flip = True

# Define the augmentation parameters
rotation_range = 60
horizontal_flip = True
vertical_flip = True

# Define the augmentation parameters
rotation_range = 45
horizontal_flip = True
vertical_flip = True

# Create an ImageDataGenerator with rotation and flip augmentations
data_gen_args = dict(
    rotation_range=rotation_range,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip
)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Function to add Gaussian noise to an image
def add_gaussian_noise(image_array, mean=0, sigma=1):
    gaussian_noise = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = image_array + gaussian_noise
    return noisy_image

# Function to add texture noise to an image with a given intensity
def add_texture_noise(image_array, intensity=0.2):
    noise = np.random.rand(*image_array.shape)
    texture_noise = intensity * noise
    noisy_image = image_array + texture_noise
    return noisy_image

# Get the list of image files in the input directory
image_files = os.listdir(input_images_dir)

# Loop through each image file
for filename in image_files:
    # Load the image
    filename_without_extension = os.path.splitext(filename)[0]
    image_path = os.path.join(input_images_dir, filename)
    image = Image.open(image_path)

    # Load the corresponding mask image
    mask_filename = filename_without_extension + '.png'  # Assume masks are in JPEG format
    mask_path = os.path.join(input_masks_dir, mask_filename)
    mask = Image.open(mask_path).convert('L')  # Convert to grayscale

    # Resize the image and mask to a consistent size
    target_size = (512, 512)
    image = image.resize(target_size)
    mask = mask.resize(target_size)

    # Convert the image and mask to numpy arrays
    image_array = np.array(image)
    mask_array = np.array(mask)

    # Expand the dimensions of the arrays to match the input shape expected by the datagen
    image_array = np.expand_dims(image_array, axis=0)
    mask_array = np.expand_dims(mask_array, axis=0)
    mask_array = np.expand_dims(mask_array, axis=-1)  # Add channel dimension

    # Generate augmented images and masks using the datagen
    seed = 1  # Set a fixed seed to ensure the same augmentation for image and mask
    augmented_images = image_datagen.flow(image_array, batch_size=1, save_to_dir=output_dir_img, save_prefix=filename_without_extension, save_format='jpg', seed=seed)
    augmented_masks = mask_datagen.flow(mask_array, batch_size=1, save_to_dir=output_dir_lab, save_prefix=filename_without_extension, save_format='png', seed=seed)

    # Generate and save augmented images and masks
    for i in range(2):  # Save 3 augmented images for each input image
        augmented_image_array = augmented_images.next()
        augmented_mask_array = augmented_masks.next()

        # Add Gaussian noise to the augmented image
        noisy_image = add_gaussian_noise(augmented_image_array[0], mean=0, sigma=1)

        # Add texture noise to the noisy image
        noisy_image_with_texture = add_texture_noise(noisy_image, intensity=0.2)

        # Save the noisy image
        noisy_image_filename = f"{filename_without_extension}_noisy_{i}.jpg"
        noisy_image_path = os.path.join(output_dir_img, noisy_image_filename)
        Image.fromarray(np.uint8(noisy_image_with_texture)).save(noisy_image_path)

        # Debugging: Print mask array shape and data type
        print(f"Mask array shape: {augmented_mask_array.shape}, dtype: {augmented_mask_array.dtype}")

        noisy_mask_filename = f"{filename_without_extension}_noisy_{i}.png"
        noisy_mask_path = os.path.join(output_dir_lab, noisy_mask_filename)

        Image.fromarray(np.uint8(augmented_mask_array[0, :, :, 0])).save(noisy_mask_path)
