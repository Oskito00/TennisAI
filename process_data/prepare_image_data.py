import json
import os
from PIL import Image
import torch
import numpy as np
from pathlib import Path

def load_tennis_dataset(json_path, images_dir):
    # Load annotations
    with open(json_path) as f:
        data = json.load(f)
    
    # Create image id to filename mapping
    image_map = {img['id']: img['file_name'] for img in data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append({
            'category_id': ann['category_id'],
            'keypoints': ann['keypoints']  # [x, y, visibility]
        })
    
    # Create final dataset structure
    dataset = []
    for image_id, annotations in annotations_by_image.items():
        image_path = Path(images_dir) / image_map[image_id]
        dataset.append({
            'image_path': str(image_path),
            'annotations': annotations
        })
    
    print("Length of dataset:", len(dataset))
    return dataset

def create_training_data(dataset_list, target_size=(224, 224)):
    """
    Convert images to RGB vectors and create corresponding labels from keypoints
    
    Args:
        dataset_list: List of dictionaries containing image paths and annotations
        target_size: Tuple of (height, width) to resize images to
    
    Returns:
        X: numpy array of shape (n_samples, height, width, 3) containing RGB values
        y: numpy array of shape (n_samples, 4) containing normalized keypoint coordinates
    """
    X = []  # Will hold image data
    Y = []  # Will hold keypoint labels
    image_info = []
    
    for data in dataset_list:
        # Load and resize image
        img_path = data['image_path']
        try:
            # Load image and convert to RGB
            img = Image.open(img_path).convert('RGB')
            # Get image dimensions
            img_width, img_height = img.size
            print(f"Image dimensions: {img_width}x{img_height}")
            print(f"Image loaded: {img_path}")
            # Resize image
            img = img.resize(target_size)
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img) / 255.0
            
            # Get keypoints (assuming 2 keypoints per image)
            keypoints = []
            for ann in data['annotations']:
                
                print("Single annotation:", ann)
                print("Keypoints type: ", type(ann['keypoints']))
                print(f"Keypoints value:", ann['keypoints'])

                kp = ann['keypoints']
                # Normalize coordinates to [0, 1]
 
                x = float(kp[0]) / img_width  # Original width 
                y = float(kp[1]) / img_height  # Original height

                keypoints.extend([x, y])
            
            if len(keypoints) == 4:  # Ensure we have 4 coordinates (x1,y1,x2,y2)
                X.append(img_array)
                Y.append(keypoints)  # Now keypoints is a list, not a float
                image_info.append({
                    'filename': Path(img_path).name,
                    'full_path': img_path
                })
            else:
                print(f"Skipping image {img_path}: found {len(keypoints)} coordinates, expected 4")

            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return np.array(X), np.array(Y), image_info

#Save the data to a file in json format
def save_data(X, y, image_info, filename):
    """Save image data and labels to CSV files"""
    import pandas as pd
    
    # Reshape the image data to 2D array (samples, pixels*channels)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    # Create DataFrame with image info and features
    features_df = pd.DataFrame(X_reshaped)
    # Add image information columns
    features_df.insert(0, 'filename', [info['filename'] for info in image_info])
    features_df.insert(1, 'full_path', [info['full_path'] for info in image_info])
    
    # Save to CSV
    features_df.to_csv(f'{filename}_features.csv', index=False)
    pd.DataFrame(y, columns=['x1', 'y1', 'x2', 'y2']).to_csv(f'{filename}_labels.csv', index=False)

# Usage example:
if __name__ == "__main__":
    json_path = '/Users/oscaralberigo/Desktop/CDING/TennisAI/coco-annotator/datasets/tennis/.exports/coco-1732187819.2898152.json'
    images_dir = '/Users/oscaralberigo/Desktop/CDING/TennisAI/coco-annotator/datasets/tennis'
    dataset = load_tennis_dataset(json_path, images_dir)
    X, y, image_info = create_training_data(dataset)
    print(f"Image data shape: {X.shape}")  # Should be (n_samples, height, width, 3)
    print(f"Label shape: {y.shape}")       # Should be (n_samples, 4)
    save_data(X, y, image_info , 'tennis_data')
