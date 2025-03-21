import os
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from typing import Tuple, List

# Configuration
CONFIG = {
    "model_path": "model/weights.keras",
    "patch_size": 32,
    "stride": 20,
    "input_dir": "input",
    "output_dir": "output",
    "batch_size": 32
}

# Create output directory
os.makedirs(CONFIG["output_dir"], exist_ok=True)

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Normalize and enhance the input image.
    
    Args:
        img: Input image array with shape (1, channels, height, width)
        
    Returns:
        Preprocessed image normalized to 0-1 range
    """
    # Check if image is already in the right format
    if img.shape[1] != 3 and img.shape[1] != 1:
        # Assume channels_last format and convert
        img = np.transpose(img, (0, 3, 1, 2))
    # Normalize
    img_norm = (img - np.mean(img)) / np.std(img)
    img_norm = (img_norm - np.min(img_norm)) / (np.max(img_norm) - np.min(img_norm)) * 255
    
    # Apply CLAHE equalization and gamma correction
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    for i in range(img.shape[1]):
        img_norm[0, i] = clahe.apply(np.array(img_norm[0, i], dtype=np.uint8))
        img_norm[0, i] = cv2.LUT(np.array(img_norm[0, i], dtype=np.uint8), table)
    
    return img_norm / 255.0

def extract_patches(img: np.ndarray, patch_size: int, stride: int) -> Tuple[np.ndarray, int, int, int, int]:
    """Extract overlapping patches from image.
    
    Args:
        img: Input image array with shape (1, channels, height, width)
        patch_size: Size of square patches to extract
        stride: Stride between patches
        
    Returns:
        Tuple containing:
        - Array of extracted patches
        - Image height (possibly padded)
        - Image width (possibly padded)
        - Number of patches in height dimension
        - Number of patches in width dimension
    """
    img_h, img_w = img.shape[2], img.shape[3]
    
    # Calculate and apply padding if needed
    pad_h = (stride - ((img_h - patch_size) % stride)) % stride
    pad_w = (stride - ((img_w - patch_size) % stride)) % stride
    
    if pad_h > 0 or pad_w > 0:
        padded_img = np.zeros((img.shape[0], img.shape[1], img_h + pad_h, img_w + pad_w))
        padded_img[:, :, :img_h, :img_w] = img
        img = padded_img
        img_h, img_w = img.shape[2], img.shape[3]
    
    # Extract patches
    n_patches_h = (img_h - patch_size) // stride + 1
    n_patches_w = (img_w - patch_size) // stride + 1
    n_patches = n_patches_h * n_patches_w
    
    patches = np.empty((n_patches, img.shape[1], patch_size, patch_size))
    patch_idx = 0
    
    for h in range(n_patches_h):
        for w in range(n_patches_w):
            patch = img[:, :, h*stride:h*stride+patch_size, w*stride:w*stride+patch_size]
            patches[patch_idx] = patch[0]
            patch_idx += 1
    
    return patches, img_h, img_w, n_patches_h, n_patches_w

def recompose_image(patches: np.ndarray, img_h: int, img_w: int, patch_size: int, 
                   stride: int, n_patches_h: int, n_patches_w: int) -> np.ndarray:
    """Recompose an image from patches with averaging in overlapping regions.
    
    Args:
        patches: Array of image patches
        img_h, img_w: Height and width of the output image
        patch_size: Size of each square patch
        stride: Stride used when extracting patches
        n_patches_h, n_patches_w: Number of patches in height and width dimensions
        
    Returns:
        Reconstructed image
    """
    reconstructed = np.zeros((1, 1, img_h, img_w))
    weights = np.zeros((1, 1, img_h, img_w))
    
    patch_idx = 0
    for h in range(n_patches_h):
        for w in range(n_patches_w):
            reconstructed[0, 0, h*stride:h*stride+patch_size, w*stride:w*stride+patch_size] += patches[patch_idx, 0]
            weights[0, 0, h*stride:h*stride+patch_size, w*stride:w*stride+patch_size] += 1
            patch_idx += 1
    
    # Average overlapping regions
    weights = np.maximum(weights, 1)  # Avoid division by zero
    reconstructed = reconstructed / weights
    
    return reconstructed

def process_predictions(predictions: np.ndarray, patch_size: int) -> np.ndarray:
    """Convert model predictions to consistent patch format.
    
    Args:
        predictions: Model output predictions
        patch_size: Size of each square patch
        
    Returns:
        Standardized patches ready for image reconstruction
    """
    pred_patches = np.empty((predictions.shape[0], 1, patch_size, patch_size))
    pred_patches = np.clip(pred_patches, 0, 1)
    
    for i in range(predictions.shape[0]):
        # Handle different prediction formats
        if len(predictions.shape) > 1 and predictions.shape[1] == 2:
            # Binary classification per patch
            pred_value = np.ones((patch_size, patch_size)) * predictions[i, 1]
            pred_patches[i, 0] = pred_value
        elif len(predictions[i].shape) > 1 and predictions[i].shape[1] == 2:
            # Binary classification for each pixel
            pred_value = predictions[i, :, 1]
            pred_patches[i, 0] = pred_value.reshape(patch_size, patch_size)
        elif predictions[i].shape[0] == patch_size * patch_size:
            # Flattened patch
            pred_patches[i, 0] = predictions[i].reshape(patch_size, patch_size)
        else:
            # Fallback option
            pred_patches[i, 0] = np.ones((patch_size, patch_size)) * np.mean(predictions[i])
    
    return pred_patches

def process_image(image_path: str, model: tf.keras.Model, patch_size: int, stride: int) -> np.ndarray:
    """Process a single image through the model pipeline.
    
    Args:
        image_path: Path to the input image
        model: Loaded TensorFlow model
        patch_size: Size of patches to extract
        stride: Stride between patches
        
    Returns:
        Prediction result as a numpy array or None if processing fails
    """
    try:
        # Load and prepare image
        image = Image.open(image_path)
        width, height = image.size
        
        # Convert to numpy array and adjust dimensions for model
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, 0)  # Add batch dimension
        if tf.keras.backend.image_data_format() == 'channels_first':
            img_array = np.transpose(img_array, (0, 3, 1, 2))
        else:
            # Keep channels_last format for CPU processing
            pass
        
        # Process image pipeline
        img_processed = preprocess_image(img_array)
        patches, img_h, img_w, n_patches_h, n_patches_w = extract_patches(
            img_processed, patch_size, stride
        )
        
        # Make predictions with configurable verbosity
        predictions = model.predict(
            patches, 
            batch_size=CONFIG["batch_size"], 
            verbose=CONFIG.get("verbose", 0)
        )
        pred_patches = process_predictions(predictions, patch_size)
        
        # Recompose the full image from patches
        reconstructed = recompose_image(
            pred_patches, img_h, img_w, patch_size, stride, n_patches_h, n_patches_w
        )
        
        # Crop to original dimensions
        return reconstructed[:, :, :height, :width]
        
    except IOError as e:
        print(f"Error reading image {image_path}: {e}")
        return None
    except ValueError as e:
        print(f"Error processing image data {image_path}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error processing {image_path}: {e}")
        return None

def save_prediction(prediction: np.ndarray, output_path: str) -> None:
    """Save prediction as a PNG image.
    
    Args:
        prediction: Prediction array
        output_path: Path to save the image (without extension)
        
    Raises:
        IOError: If the image cannot be saved
    """
    if prediction is None:
        return
        
    try:
        # Convert to 0-255 range
        img = (prediction[0, 0] * 255).astype(np.uint8)
        
        # Use os.path.join for path handling and ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save using PIL with proper path construction
        Image.fromarray(img).save(f"{output_path}.png")
    except IOError as e:
        print(f"Error saving image to {output_path}: {e}")
    except Exception as e:
        print(f"Unexpected error saving to {output_path}: {e}")

def main():
    """Main processing function"""
    # Load model
    try:
        model = tf.keras.models.load_model(CONFIG["model_path"])
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get list of input files
    input_files = [f for f in os.listdir(CONFIG["input_dir"]) 
                  if os.path.isfile(os.path.join(CONFIG["input_dir"], f)) 
                  and not f.startswith('.')]
    
    if not input_files:
        print(f"No files found in {CONFIG['input_dir']} directory")
        return
        
    # Process all images
    for filename in input_files:
        print(f"Processing {filename}...")
        path_image = os.path.join(CONFIG["input_dir"], filename)
        
        # Process the image
        prediction = process_image(path_image, model, CONFIG["patch_size"], CONFIG["stride"])
        
        # Save the prediction
        if prediction is not None:
            base_name = os.path.splitext(filename)[0]
            output_filename = os.path.join(CONFIG["output_dir"], f"{base_name}_pred")
            save_prediction(prediction, output_filename)
    
    print("All done! Please check the output folder")

if __name__ == '__main__':
    main()