import json
import os
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd

def load_cropped_tennis_dataset(json_path, images_dir):
    """Load the cropped tennis dataset with both keypoints."""
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
            annotations_by_image[image_id] = {}
        
        # Store keypoints by category (1=top, 2=bottom)
        category_id = ann['category_id']
        if ann['keypoints'][2] > 0:  # Only store visible keypoints
            annotations_by_image[image_id][category_id] = ann['keypoints'][:2]  # Store only x,y
    
    # Create final dataset structure
    dataset = []
    for image_id, annotations in annotations_by_image.items():
        # Only include images that have both keypoints
        if 1 in annotations and 2 in annotations:
            image_path = Path(images_dir) / image_map[image_id]
            dataset.append({
                'image_path': str(image_path),
                'top_keypoint': annotations[1],    # category 1
                'bottom_keypoint': annotations[2]  # category 2
            })
    
    print(f"Found {len(dataset)} valid images with both keypoints")
    return dataset

def create_training_data(dataset_list, target_size=(224, 224)):
    """Convert images to RGB vectors and create corresponding labels."""
    X = []  # Image data
    Y = []  # Keypoint coordinates [x1,y1,x2,y2]
    image_info = []  # Image metadata
    
    for data in dataset_list:
        try:
            # Load and process image
            img = Image.open(data['image_path']).convert('RGB')
            img_width, img_height = img.size
            
            # Resize image
            img = img.resize(target_size)
            img_array = np.array(img) / 255.0
            
            # Get normalized keypoints
            top_x, top_y = data['top_keypoint']
            bottom_x, bottom_y = data['bottom_keypoint']
            
            # Normalize coordinates
            keypoints = [
                float(top_x) / img_width,
                float(top_y) / img_height,
                float(bottom_x) / img_width,
                float(bottom_y) / img_height
            ]
            
            X.append(img_array)
            Y.append(keypoints)
            image_info.append({
                'filename': Path(data['image_path']).name,
                'full_path': data['image_path']
            })
            
        except Exception as e:
            print(f"Error processing {data['image_path']}: {e}")
            continue
    
    return np.array(X), np.array(Y), image_info

def save_data(X, y, image_info, output_dir):
    """Save image data and labels to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Reshape the image data to 2D array (samples, pixels*channels)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    # Create DataFrame with image info and features
    features_df = pd.DataFrame(X_reshaped)
    features_df.insert(0, 'filename', [info['filename'] for info in image_info])
    features_df.insert(1, 'full_path', [info['full_path'] for info in image_info])
    
    # Create labels DataFrame
    labels_df = pd.DataFrame(y, columns=['top_x', 'top_y', 'bottom_x', 'bottom_y'])
    
    # Save to CSV
    features_path = os.path.join(output_dir, 'cropped_features.csv')
    labels_path = os.path.join(output_dir, 'cropped_labels.csv')
    
    features_df.to_csv(features_path, index=False)
    labels_df.to_csv(labels_path, index=False)
    
    print(f"Saved features to: {features_path}")
    print(f"Saved labels to: {labels_path}")

if __name__ == "__main__":
    # Update these paths to match your setup
    json_path = "coco-annotator/datasets/tennis_cropped/cropped_annotations.json"
    images_dir = "coco-annotator/datasets/tennis_cropped"
    output_dir = "process_data/cropped_data"
    
    # Load and process dataset
    dataset = load_cropped_tennis_dataset(json_path, images_dir)
    X, y, image_info = create_training_data(dataset)
    
    print(f"Processed data shapes:")
    print(f"Images (X): {X.shape}")  # Should be (n_samples, height, width, 3)
    print(f"Labels (y): {y.shape}")  # Should be (n_samples, 4)
    
    # Save the processed data
    save_data(X, y, image_info, output_dir) 