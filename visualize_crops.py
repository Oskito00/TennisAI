import os
import json
import cv2
import numpy as np

def draw_keypoints(image_path, annotations, output_path):
    """Draw all keypoints on image and save it."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Print annotations for debugging
    print(f"Annotations for {image_path}:", annotations)
    
    for ann in annotations:
        keypoints = ann['keypoints']
        category_id = ann['category_id']
        
        # Skip if not category 1 or 2
        if category_id not in [1, 2]:
            continue
            
        x = int(keypoints[0])
        y = int(keypoints[1])
        visibility = keypoints[2]
        
        if visibility > 0:  # Only draw visible keypoints
            # Different colors for top and bottom
            if category_id == 1:  # Top point
                color = (0, 0, 255)  # Red in BGR
                label = "Top"
            elif category_id == 2:  # Bottom point
                color = (255, 0, 0)  # Blue in BGR
                label = "Bottom"
                
            # Draw keypoint
            cv2.circle(img, (x, y), 5, color, -1)  # Filled circle
            cv2.circle(img, (x, y), 7, color, 2)   # Circle outline
            
            # Add label
            cv2.putText(img, label, (x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the annotated image
    cv2.imwrite(output_path, img)
    print(f"Saved annotated image: {output_path}")

# Load the cropped annotations
cropped_coco_file = "coco-annotator/datasets/tennis_cropped/cropped_annotations.json"
with open(cropped_coco_file, 'r') as f:
    coco_data = json.load(f)

# Create output directory for visualizations
vis_dir = "coco-annotator/datasets/tennis_cropped/visualizations"
os.makedirs(vis_dir, exist_ok=True)

# Create mapping from image_id to filename and annotations
image_map = {img['id']: img['file_name'] for img in coco_data['images']}
annotations_by_image = {}

# Group annotations by image_id
for ann in coco_data['annotations']:
    if 'keypoints' in ann:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

# Process each image
for image_id, filename in image_map.items():
    if image_id in annotations_by_image:
        # Construct paths
        image_path = os.path.join("coco-annotator/datasets/tennis_cropped", filename)
        output_path = os.path.join(vis_dir, f"vis_{filename}")
        
        # Draw and save visualization with all keypoints
        draw_keypoints(image_path, annotations_by_image[image_id], output_path)

print("Visualization complete! Check the 'visualizations' directory.") 