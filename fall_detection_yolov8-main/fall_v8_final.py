from ultralytics import YOLO
import numpy as np
import cv2
import math
model = YOLO("yolov8n-pose.pt") 

# Define keypoint labels for your pose model (order should match the model's output)
labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
          "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
          "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

def fall_detection(poses):
    """
    Detects fall events based on pose information.
    
    :param poses: List of poses, where each pose is a list of body part coordinates.
                  Format of pose: [index, confidence, x_center, y_center, width, height, ...body parts...]
    :return: Tuple (fall_detected, bounding_box). 
             fall_detected: True if a fall is detected, False otherwise.
             bounding_box: (xmin, ymin, xmax, ymax) of the fall region if a fall is detected.
    """
    bounding_boxes = []
    for pose in poses:
        bbox = pose['bbox']
        keypoint = pose['keypoints']

        x,y,x1,y1 = np.array(bbox).astype('int')
        left_shoulder_x, left_shoulder_y = keypoint[5][0], keypoint[5][1]
        right_shoulder_y = keypoint[6][1]
        left_body_x, left_body_y = keypoint[11][0], keypoint[11][1]
        right_body_y = keypoint[12][1]
        left_foot_y, right_foot_y = keypoint[15][1], keypoint[16][1]
        nose_x,nose_y = keypoint[0][0],keypoint[0][1]

        # Calculate the distance between shoulder and body as a factor
        len_factor = math.sqrt((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2)

        # Calculate bounding box dimensions
        dx = x1 - x
        dy = y1 - y

        # Check conditions for a fall
        difference = dx /dy
        fall_condition = (
            # Left side condition
            ((
                left_shoulder_y > left_foot_y - len_factor and
                left_body_y > left_foot_y - (len_factor / 2) and
                left_shoulder_y > left_body_y - (len_factor / 2)
            ) or
            # Right side condition
            (
                right_shoulder_y > right_foot_y - len_factor and
                right_body_y > right_foot_y - (len_factor / 2) and
                right_shoulder_y > right_body_y - (len_factor / 2)
            )) and ((nose_x and nose_y) >0.0 ) or
           
            (difference > 1.4 
        ))
        if fall_condition:
            bounding_boxes.append([True,[x, y, x1, y1]])
    return bounding_boxes

cap = cv2.VideoCapture("fall1.mp4")
# Assuming 'results' is the output from the model
width = int(cap.get(3))
height = int(cap.get(4))
size=(width,height)
output = cv2.VideoWriter('fall_output.avi',cv2.VideoWriter_fourcc(*'MPEG'),30,size)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(frame,conf=0.7)
    detections = []
    for result in results:
        bbox = result.boxes.xyxy.tolist()  # get the bounding box on the frame
        confidence = result.boxes.conf.tolist() # get the confident it is a human from a frame
        keypoints = result.keypoints.xyn.tolist() # get the every human keypoint from a frame
        
        for box,keypoint in zip(bbox,keypoints):
            detections.append({'bbox':box,'keypoints':keypoint})
        
        result_fall = fall_detection(detections)
        for res in result_fall:
            ret,bbox = res
            if ret:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)
                cv2.putText(frame,"Fall detected", (x1,y1+50),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
        output.write(frame)
output.release()
cap.release()

