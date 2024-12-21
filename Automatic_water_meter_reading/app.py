import os
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models


# Flask app initialization
app = Flask(__name__)

# Set the upload folder path
UPLOAD_FOLDER = "./upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def run_inference(image):
    # Preprocess the image (convert to tensor)
    image_tensor = transform(image).unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        prediction = model_rcnn(image_tensor)
    # Extract predictions
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    return boxes, labels, scores

def draw_boxes(image, boxes, labels, scores, threshold=0.9):
    # Draw bounding boxes on the image
    image_copy = np.array(image).copy()
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            x_min, y_min, x_max, y_max = box
            label = labels[i]
            confidence = scores[i]
            # Draw the rectangle and label
            cv2.rectangle(image_copy, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(image_copy, f'{label}', (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image_copy



# Load the YOLOv8 model
model = YOLO('yolov8m-seg.pt')
num_classes = 9 
# Load the pre-trained Faster R-CNN model with a ResNet backbone
model_rcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model_rcnn.roi_heads.box_predictor.cls_score.in_features
model_rcnn.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
transform = transforms.Compose([
    transforms.ToTensor(),
])
checkpoint = torch.load('./faster_rcnn_model_new.pth')
model_rcnn.load_state_dict(checkpoint) 
model_rcnn.eval()


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html') 

# Route for file upload and inference
@app.route('/upload', methods=['POST'])
def upload_files():
    meter_detected = False
    if 'files[]' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files[]')
    if not files:
        return jsonify({"error": "No files selected"}), 400

    for file in files:
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the image
        image = cv2.imread(file_path)
        results = model(image)  # Run meter detection model

        for res in results:       
    
            boxes = res.boxes.cpu().numpy()  # get boxes in numpy
                
            for box_lpd in boxes:  # iterate boxes
                r1 = box_lpd.xyxy[0].astype(int)   
                x2,y2,x3,y3 = r1   
                w = x3 - x2
                h = y3 -y2
                ocr = image[y2:y3,x2:x3]
                cv2.imwrite('temp.png',ocr)
                meter_detected = True  # Mark detection as successful
                break        
        
        if not meter_detected:
            return jsonify({"error": "No valid meter detected. Please try again with a different image."}), 400
        else:
            # OCR Inference
            ocr_image = Image.open('temp.png') 
            boxes, labels, scores = run_inference(ocr_image)
            # Draw bounding boxes
            result_image = draw_boxes(ocr_image, boxes, labels, scores, threshold=0.5)
            result_image_pil = Image.fromarray(result_image)
            result_image_path = os.path.join(UPLOAD_FOLDER, 'result.png')
            result_image_pil.save(result_image_path)

            return send_file(result_image_path, mimetype='image/png', as_attachment=False)
    
    return jsonify({"message": "Inference complete!"}), 200

# Run the app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
