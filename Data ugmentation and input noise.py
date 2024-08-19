import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths to your input and output directories with validation
def validate_directory(directory_path):
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist. Please check the path and try again.")
    return directory_path

input_images_dir = validate_directory('/content/Deepcrack_2%/train/image/')
input_masks_dir = validate_directory('/content/Deepcrack_2%/train/label/')
output_dir_img = '/content/train/image/'
output_dir_lab = '/content/train/label/'

# Ensure output directories exist, with error handling
def create_output_directory(directory_path):
    try:
        os.makedirs(directory_path, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create directory {directory_path}: {e}")

create_output_directory(output_dir_img)
create_output_directory(output_dir_lab)

# Augmentation parameter definitions with verbosity
def define_augmentation_params(rotation_range=45, horizontal_flip=True, vertical_flip=True):
    print(f"Defining augmentation parameters with rotation_range={rotation_range}, "
          f"horizontal_flip={horizontal_flip}, vertical_flip={vertical_flip}")
    return dict(
        rotation_range=rotation_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip
    )

# Create ImageDataGenerators with detailed logs
data_gen_args = define_augmentation_params(rotation_range=90)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Function to add Gaussian noise with error checking
def add_gaussian_noise(image_array, mean=0, sigma=1):
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input image_array must be a numpy array")
    gaussian_noise = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = np.clip(image_array + gaussian_noise, 0, 255)  # Ensure pixel values are within valid range
    return noisy_image

# Function to add texture noise with optional intensity adjustment
def add_texture_noise(image_array, intensity=0.2):
    if not isinstance(image_array, np.ndarray):
        raise TypeError("Input image_array must be a numpy array")
    noise = np.random.rand(*image_array.shape)
    texture_noise = intensity * noise
    noisy_image = np.clip(image_array + texture_noise, 0, 255)  # Ensure pixel values are within valid range
    return noisy_image

# Get the list of image files with validation
def list_files(directory):
    files = os.listdir(directory)
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")
    return files

image_files = list_files(input_images_dir)

# Loop through each image file with extensive logging
for filename in tqdm(image_files, desc="Processing images"):
    # Load the image with error handling
    try:
        filename_without_extension = os.path.splitext(filename)[0]
        image_path = os.path.join(input_images_dir, filename)
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to load image {filename}: {e}")
        continue

    # Load the corresponding mask image with error handling
    try:
        mask_filename = filename_without_extension + '.png'  # Assume masks are in PNG format
        mask_path = os.path.join(input_masks_dir, mask_filename)
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
    except Exception as e:
        print(f"Failed to load mask for {filename}: {e}")
        continue

    # Resize the image and mask to a consistent size with error handling
    try:
        target_size = (512, 512)
        image = image.resize(target_size)
        mask = mask.resize(target_size)
    except Exception as e:
        print(f"Failed to resize image or mask for {filename}: {e}")
        continue

    # Convert the image and mask to numpy arrays with validation
    try:
        image_array = np.array(image)
        mask_array = np.array(mask)
    except Exception as e:
        print(f"Failed to convert image or mask to array for {filename}: {e}")
        continue

    # Expand dimensions for augmentation compatibility
    image_array = np.expand_dims(image_array, axis=0)
    mask_array = np.expand_dims(mask_array, axis=0)
    mask_array = np.expand_dims(mask_array, axis=-1)  # Add channel dimension for masks

    # Generate augmented images and masks with detailed seed handling
    seed = random.randint(1, 10000)  # Randomized seed for variability in augmentations
    print(f"Using seed {seed} for image {filename} augmentation")
    augmented_images = image_datagen.flow(image_array, batch_size=1, save_to_dir=output_dir_img, 
                                          save_prefix=filename_without_extension, save_format='jpg', seed=seed)
    augmented_masks = mask_datagen.flow(mask_array, batch_size=1, save_to_dir=output_dir_lab, 
                                        save_prefix=filename_without_extension, save_format='png', seed=seed)

    # Generate and save augmented images and masks with robust error handling
    for i in range(2):  # Save 2 augmented images for each input image
        try:
            augmented_image_array = augmented_images.next()
            augmented_mask_array = augmented_masks.next()

            # Add Gaussian noise with logging
            noisy_image = add_gaussian_noise(augmented_image_array[0], mean=0, sigma=1)
            noisy_image_with_texture = add_texture_noise(noisy_image, intensity=0.2)

            # Save the noisy image with detailed logging
            noisy_image_filename = f"{filename_without_extension}_noisy_{i}.jpg"
            noisy_image_path = os.path.join(output_dir_img, noisy_image_filename)
            Image.fromarray(np.uint8(noisy_image_with_texture)).save(noisy_image_path)
            print(f"Saved noisy image {noisy_image_filename} to {output_dir_img}")

            # Save the noisy mask with logging
            noisy_mask_filename = f"{filename_without_extension}_noisy_{i}.png"
            noisy_mask_path = os.path.join(output_dir_lab, noisy_mask_filename)
            Image.fromarray(np.uint8(augmented_mask_array[0, :, :, 0])).save(noisy_mask_path)
            print(f"Saved noisy mask {noisy_mask_filename} to {output_dir_lab}")

            # Debugging: Print mask array shape and data type for further inspection
            print(f"Mask array shape: {augmented_mask_array.shape}, dtype: {augmented_mask_array.dtype}")
        except Exception as e:
            print(f"Failed during augmentation or saving for {filename}: {e}")
            continue
