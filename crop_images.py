import os
import json
from PIL import Image

# Load the COCO file with bounding box annotations
coco_file = "coco-annotator/datasets/tennis/.exports/coco-1732187819.2898152_with_bbox.json"
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Create output directory for cropped images
output_dir = "coco-annotator/datasets/tennis_cropped"
os.makedirs(output_dir, exist_ok=True)

# Margin percentage
MARGIN = 0.2

# Create dictionaries for mapping
image_to_bbox = {}
image_to_keypoints = {}

# Map bboxes and keypoints to image IDs
for ann in coco_data['annotations']:
    image_id = ann['image_id']
    
    if 'bbox' in ann:
        if image_id not in image_to_bbox:
            image_to_bbox[image_id] = []
        image_to_bbox[image_id].append(ann['bbox'])
    
    if 'keypoints' in ann:
        if image_id not in image_to_keypoints:
            image_to_keypoints[image_id] = []
        image_to_keypoints[image_id].append(ann['keypoints'])

# Prepare new COCO data structure for cropped images
new_coco_data = {
    'images': [],
    'annotations': [],
    'categories': coco_data['categories']
}

new_image_id = 1
new_annotation_id = 1

# Process each image
for image_info in coco_data['images']:
    image_id = image_info['id']
    
    # Skip if no bbox annotations
    if image_id not in image_to_bbox:
        continue
        
    filename = image_info['file_name']
    image_path = os.path.join("coco-annotator/datasets/tennis", filename)
    
    if not os.path.exists(image_path):
        continue
        
    try:
        img = Image.open(image_path)
        
        # Process each bbox for this image
        for i, bbox in enumerate(image_to_bbox[image_id]):
            x, y, w, h = bbox
            
            # Calculate margins and crop coordinates
            margin_w = w * MARGIN
            margin_h = h * MARGIN
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(img.width, x + w + margin_w)
            y2 = min(img.height, y + h + margin_h)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop image
            cropped = img.crop((x1, y1, x2, y2))
            
            if cropped.size[0] <= 0 or cropped.size[1] <= 0:
                continue
            
            # Generate output filename
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_crop{i}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save cropped image
            cropped.save(output_path, "JPEG")
            
            # Add new image entry
            new_image_info = {
                'id': new_image_id,
                'file_name': output_filename,
                'width': cropped.size[0],
                'height': cropped.size[1]
            }
            new_coco_data['images'].append(new_image_info)
            
            # Inside the main loop, in the keypoint processing section:
            if image_id in image_to_keypoints:
                # Group keypoints by their category_id
                keypoints_by_category = {}
                for ann in coco_data['annotations']:
                    if ann['image_id'] == image_id and 'keypoints' in ann:
                        print("Found annotation with category_id:", ann['category_id'])  # Debug print
                        category_id = ann['category_id']
                        if category_id not in keypoints_by_category:
                            keypoints_by_category[category_id] = []
                        keypoints_by_category[category_id].append({
                            'keypoints': ann['keypoints'],
                            'category_id': category_id  # Store the category_id with the keypoints
                        })
                
                # Process each category's keypoints
                for category_id, keypoints_list in keypoints_by_category.items():
                    for kp_data in keypoints_list:
                        keypoints = kp_data['keypoints']
                        original_category_id = kp_data['category_id']  # Get the original category_id
                        
                        # Convert keypoints to new coordinate system
                        new_keypoints = []
                        for j in range(0, len(keypoints), 3):
                            kp_x = keypoints[j]
                            kp_y = keypoints[j + 1]
                            kp_v = keypoints[j + 2]
                            
                            # Only adjust if keypoint is visible
                            if kp_v > 0:
                                # Check if keypoint is within crop bounds
                                if x1 <= kp_x <= x2 and y1 <= kp_y <= y2:
                                    new_keypoints.extend([
                                        kp_x - x1,  # Adjust x coordinate
                                        kp_y - y1,  # Adjust y coordinate
                                        kp_v        # Keep visibility unchanged
                                    ])
                                else:
                                    new_keypoints.extend([0, 0, 0])  # Keypoint outside crop
                            else:
                                new_keypoints.extend([0, 0, 0])
                        
                        # Add new annotation with correct category_id
                        new_annotation = {
                            'id': new_annotation_id,
                            'image_id': new_image_id,
                            'category_id': original_category_id,  # Use the original category_id
                            'keypoints': new_keypoints,
                            'num_keypoints': sum(1 for i in range(2, len(new_keypoints), 3) if new_keypoints[i] > 0)
                        }
                        new_coco_data['annotations'].append(new_annotation)
                        new_annotation_id += 1
            
            new_image_id += 1
            
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# Save new COCO file
output_coco_file = os.path.join(output_dir, "cropped_annotations.json")
with open(output_coco_file, 'w') as f:
    json.dump(new_coco_data, f)

print("Cropping complete!")
print(f"New annotations saved to: {output_coco_file}")