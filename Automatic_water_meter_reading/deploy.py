#import necessary libraries
import streamlit as st
import os
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models

# Display the logo at the top of the app
st.set_page_config(
    page_title="Utility Meter Reading",
    layout="centered"
)
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

# Set the title of the Streamlit app
st.title("Automatic Utility Meter Reading")


# Set the upload folder path
UPLOAD_FOLDER = "./upload"
meter_detected = False 
# Section for uploading data for training
st.header("Upload Data for Inference")
uploaded_files = st.file_uploader("Upload your data files", accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files to the server (streamlit folder)
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        # Create directories if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Get actual folder path of the upload directory
    actual_path = UPLOAD_FOLDER
    st.success(f"Files successfully uploaded to {actual_path}")
    # Run inference on the uploaded files (images)
    for uploaded_file in uploaded_files:
        image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        # Perform inference
        image = cv2.imread(image_path)
        
        results = model(image)  # Inference
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
    
            st.error("No valid meter detected. Please try again with a different image.")
        else:
    
            ocr_image = Image.open('temp.png') 
            # Run inference
            boxes, labels, scores = run_inference(ocr_image)
            
            # Draw bounding boxes on the image
            result_image = draw_boxes(ocr, boxes, labels, scores, threshold=0.5)
            
            # Convert result image to display in Streamlit
            result_image_pil = Image.fromarray(result_image)
            
            # Display result
            st.image(result_image_pil, caption="Inference Result", use_container_width=True)

            st.write("Inference complete!")

        
else:
    st.info("Please upload some data files.")
