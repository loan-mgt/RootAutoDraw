import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

# Create output directory
os.makedirs("output", exist_ok=True)

def preprocess_image(img):
    """Simple preprocessing pipeline for the image"""
    # Normalize
    img_norm = (img - np.mean(img)) / np.std(img)
    img_norm = (img_norm - np.min(img_norm)) / (np.max(img_norm) - np.min(img_norm)) * 255
    
    # Apply CLAHE equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(img.shape[1]):
        img_norm[0, i] = clahe.apply(np.array(img_norm[0, i], dtype=np.uint8))
    
    # Adjust gamma
    gamma = 1.2
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    for i in range(img.shape[1]):
        img_norm[0, i] = cv2.LUT(np.array(img_norm[0, i], dtype=np.uint8), table)
    
    # Scale to 0-1
    return img_norm / 255.0

def extract_patches(img, patch_h, patch_w, stride_h, stride_w):
    """Extract overlapping patches from image"""
    img_h, img_w = img.shape[2], img.shape[3]
    
    # Calculate padding if needed
    pad_h = 0
    pad_w = 0
    if (img_h - patch_h) % stride_h != 0:
        pad_h = stride_h - ((img_h - patch_h) % stride_h)
    if (img_w - patch_w) % stride_w != 0:
        pad_w = stride_w - ((img_w - patch_w) % stride_w)
    
    # Pad the image if needed
    if pad_h > 0 or pad_w > 0:
        padded_img = np.zeros((img.shape[0], img.shape[1], img_h + pad_h, img_w + pad_w))
        padded_img[:, :, :img_h, :img_w] = img
        img = padded_img
        img_h, img_w = img.shape[2], img.shape[3]
    
    # Calculate number of patches
    n_patches_h = (img_h - patch_h) // stride_h + 1
    n_patches_w = (img_w - patch_w) // stride_w + 1
    n_patches = n_patches_h * n_patches_w
    
    # Extract patches
    patches = np.empty((n_patches, img.shape[1], patch_h, patch_w))
    patch_idx = 0
    for h in range(n_patches_h):
        for w in range(n_patches_w):
            patch = img[:, :, h*stride_h:h*stride_h+patch_h, w*stride_w:w*stride_w+patch_w]
            patches[patch_idx] = patch[0]  # Take first image in batch
            patch_idx += 1
    
    return patches, img_h, img_w, n_patches_h, n_patches_w

def recompose_image(patches, img_h, img_w, patch_h, patch_w, stride_h, stride_w, n_patches_h, n_patches_w):
    """Recompose an image from patches with averaging in overlapping regions"""
    reconstructed = np.zeros((1, 1, img_h, img_w))  # Assuming single channel output
    weights = np.zeros((1, 1, img_h, img_w))
    
    patch_idx = 0
    for h in range(n_patches_h):
        for w in range(n_patches_w):
            reconstructed[0, 0, h*stride_h:h*stride_h+patch_h, w*stride_w:w*stride_w+patch_w] += patches[patch_idx, 0]
            weights[0, 0, h*stride_h:h*stride_h+patch_h, w*stride_w:w*stride_w+patch_w] += 1
            patch_idx += 1
    
    # Average overlapping regions
    reconstructed = reconstructed / weights
    return reconstructed

def process_image(image_path, model, patch_size, stride):
    """Process a single image through the model pipeline"""
    # Load and prepare image
    image = Image.open(image_path)
    width, height = image.size
    
    # Convert to numpy array and adjust dimensions for model
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, 0)  # Add batch dimension
    img_array = np.transpose(img_array, (0, 3, 1, 2))  # Channels first
    
    # Preprocess image
    img_processed = preprocess_image(img_array)
    
    # Extract patches
    patches, img_h, img_w, n_patches_h, n_patches_w = extract_patches(
        img_processed, patch_size, patch_size, stride, stride
    )
    
    # Make predictions
    predictions = model.predict(patches, batch_size=32, verbose=0)
    
    # Convert predictions to image patches
    pred_patches = np.empty((predictions.shape[0], 1, patch_size, patch_size))
    for i in range(predictions.shape[0]):
        # Check prediction shape to handle different model outputs
        if len(predictions.shape) > 1 and predictions.shape[1] == 2:
            # Model outputs [batch_size, 2] - binary classification per patch
            pred_value = np.ones((patch_size, patch_size)) * predictions[i, 1]
            pred_patches[i, 0] = pred_value
        elif len(predictions[i].shape) > 1 and predictions[i].shape[1] == 2:
            # Model outputs [batch_size, N, 2] - binary classification for N pixels
            pred_value = predictions[i, :, 1]
            # Reshape the flattened prediction back to patch dimensions
            pred_patches[i, 0] = pred_value.reshape(patch_size, patch_size)
        elif predictions[i].shape[0] == patch_size * patch_size:
            # Model outputs flattened patch
            pred_patches[i, 0] = predictions[i].reshape(patch_size, patch_size)
        else:
            # Just try to reshape as best we can
            pred_patches[i, 0] = np.ones((patch_size, patch_size)) * np.mean(predictions[i])
    
    # Recompose the full image from patches
    reconstructed = recompose_image(
        pred_patches, img_h, img_w, patch_size, patch_size, 
        stride, stride, n_patches_h, n_patches_w
    )
    
    # Crop to original dimensions
    reconstructed = reconstructed[:, :, :height, :width]
    
    return reconstructed

def save_prediction(prediction, output_path):
    """Save prediction as an image"""
    # Convert to 0-255 range
    img = (prediction[0, 0] * 255).astype(np.uint8)
    
    # Save using PIL
    Image.fromarray(img).save(output_path + '.png')

if __name__ == '__main__':
    # Parameters
    model_path = "model/weights.keras"
    patch_size = 32
    stride = 20
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Process all images in input directory
    for filename in listdir("input"):
        if filename.startswith('.'):  # Skip hidden files
            continue
            
        print(f"Processing {filename}...")
        path_image = join("input", filename)
        
        # Process the image
        prediction = process_image(path_image, model, patch_size, stride)
        
        # Save the prediction
        output_filename = join("output", filename[:-4] + "_pred")
        save_prediction(prediction, output_filename)
    
    print("All done! Please check the output folder")