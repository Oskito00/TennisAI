import os
import json
from inference_sdk import InferenceHTTPClient

# Initialize the HTTP client with your API key
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="MJs5ZxrQEx2LMEJVUmWd"
)

# Load existing COCO file
coco_file = "coco-annotator/datasets/tennis/.exports/coco-1732187819.2898152.json"
with open(coco_file, 'r') as f:
    coco_data = json.load(f)

# Get list of images that have keypoint annotations
annotated_image_ids = set()
for ann in coco_data['annotations']:
    if 'keypoints' in ann:  # Only include images with keypoint annotations
        annotated_image_ids.add(ann['image_id'])

# Keep track of annotation ID
annotation_id = max([ann['id'] for ann in coco_data['annotations']], default=0) + 1

# Initialize bbox annotations category if it doesn't exist
if not any(cat['name'] == 'tennis_racket' for cat in coco_data['categories']):
    coco_data['categories'].append({'id': len(coco_data['categories']) + 1, 'name': 'tennis_racket'})

# Process only annotated images
for image in coco_data['images']:
    if image['id'] not in annotated_image_ids:
        continue
        
    filename = image['file_name']
    image_path = os.path.join('coco-annotator/datasets/tennis', filename)
    
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        
        # Perform inference
        result = CLIENT.infer(image_path, model_id="tennis-racket-detection-qbwtm/2")
        
        # Add predictions as annotations
        for pred in result['predictions']:
            x = pred['x']
            y = pred['y']
            width = pred['width']
            height = pred['height']
            
            # Convert to COCO format (x,y,width,height)
            bbox = [
                x - width/2,  # x_min
                y - height/2, # y_min
                width,       # width
                height      # height
            ]
            
            annotation = {
                'id': annotation_id,
                'image_id': image['id'],
                'category_id': next(cat['id'] for cat in coco_data['categories'] if cat['name'] == 'tennis_racket'),
                'bbox': bbox,
                'area': width * height,
                'iscrowd': 0
            }
            
            coco_data['annotations'].append(annotation)
            annotation_id += 1

# Save updated COCO file
output_file = coco_file.replace('.json', '_with_bbox.json')
with open(output_file, 'w') as f:
    json.dump(coco_data, f)

print(f"Updated COCO file saved to: {output_file}")