import torch
import json
import cv2
from ultralytics import YOLO
import numpy as np
import os
from tqdm import tqdm
# Load the YOLOv5 model
model = YOLO('yolov8n.pt')
output_video_path = 'output_video.mp4'
# Load an image
img_folder = 'frames/'
image_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.png')]
image_files.sort()
# print(image_files)
first_image = cv2.imread(image_files[0])
height, width, layers = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_video_path, fourcc, 4.0, (width, height))

output_file = 'person_boxes.json'

megabox = []
# Extract bounding box coordinates for detected persons
for i, image_file in tqdm(enumerate(image_files)):
    image = cv2.imread(image_file)
        
        # Run inference
    results = model.predict(image_file)
    person_boxes = []
    for result in results:
        for detection in result.boxes:
            if int(detection.cls) == 0:  # Assuming class ID 0 corresponds to 'person'
                x = detection.xyxy
                x1, y1, x2, y2 = map(int, x.cpu().detach().numpy().tolist()[0])
                # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                person_boxes.append({
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    megabox.append(person_boxes)
    video_writer.write(image)

# Save the bounding boxes to a JSON file

with open(output_file, 'w') as f:
    json.dump(megabox, f, indent=2)
video_writer.release()

print(f"Video saved at {output_video_path}")
output_image_path = 'output_image.jpg'