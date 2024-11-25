import numpy as np
import pywt
import pandas as pd
from PIL import Image
from pathlib import Path
import os

def extract_wavelet_features(img_array, wavelet='db1', level=2, n_features=14885):
    """
    Extract wavelet features from an image using 2D discrete wavelet transform.
    
    Args:
        img_array: RGB image array (height, width, 3)
        wavelet: Wavelet family to use (default: 'db1' - Daubechies 1)
        level: Decomposition level (default: 2)
        n_features: Number of features to return (default: 14885)
    
    Returns:
        features: Fixed-length array of wavelet coefficients
    """
    print(f"\nExtracting wavelet features using {wavelet} wavelet at level {level}")
    
    # Convert to grayscale by taking mean across channels
    gray_img = np.mean(img_array, axis=2)
    print(f"Converted RGB image to grayscale with shape: {gray_img.shape}")
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)
    print(f"Performed {level}-level wavelet decomposition")
    
    # Extract all coefficients without thresholding
    features = []
    for i, coeff_list in enumerate(coeffs):
        if isinstance(coeff_list, tuple):
            print(f"Level {i} detail coefficients:")
            for j, detail_coeff in enumerate(coeff_list):
                print(f"  Detail {j} shape: {detail_coeff.shape}")
                features.extend(detail_coeff.flatten())
        else:
            print(f"Approximation coefficients shape: {coeff_list.shape}")
            features.extend(coeff_list.flatten())
    
    features = np.array(features)
    print(f"Total extracted features before selection: {len(features)}")
    
    # Sort by magnitude and select top n_features
    indices = np.argsort(np.abs(features))[-n_features:]
    features = features[indices]
    print(f"Selected top {n_features} features by magnitude")
    
    return features

def process_dataset_with_wavelets(features_path, output_dir):
    """
    Process the existing dataset to create wavelet-based features.
    
    Args:
        features_path: Path to the existing features CSV
        output_dir: Directory to save the wavelet features
    """
    print("\n=== Starting Wavelet Feature Extraction ===")
    
    # Load existing dataset
    print(f"\nLoading dataset from: {features_path}")
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} images to process")
    
    # Extract image paths and reshape original features back to images
    image_paths = df['full_path'].values
    filenames = df['filename'].values
    
    # Process each image
    wavelet_features_list = []
    processed_files = []
    
    print("\nProcessing images...")
    for i, img_path in enumerate(image_paths):
        try:
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))  # Match original preprocessing
            img_array = np.array(img) / 255.0
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {Path(img_path).name}")
            print(f"Image shape after preprocessing: {img_array.shape}")
            
            # Extract wavelet features
            features = extract_wavelet_features(img_array)
            wavelet_features_list.append(features)
            processed_files.append(Path(img_path).name)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print("\nCreating feature DataFrame...")
    # Create DataFrame with wavelet features
    wavelet_features_array = np.vstack(wavelet_features_list)
    feature_columns = [f'wavelet_{i}' for i in range(wavelet_features_array.shape[1])]
    
    wavelet_df = pd.DataFrame(wavelet_features_array, columns=feature_columns)
    wavelet_df.insert(0, 'filename', processed_files)
    
    # Save wavelet features
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'wavelet_features.csv')
    wavelet_df.to_csv(output_path, index=False)
    
    print(f"\nSaved wavelet features to: {output_path}")
    print(f"Final wavelet features shape: {wavelet_features_array.shape}")
    print(f"Successfully processed {len(processed_files)} images")
    print("\n=== Wavelet Feature Extraction Complete ===")

if __name__ == "__main__":
    # Update these paths to match your setup
    features_path = "process_data/cropped_data/cropped_features.csv"
    output_dir = "process_data/cropped_data"
    
    # Process dataset with wavelets
    process_dataset_with_wavelets(features_path, output_dir) 