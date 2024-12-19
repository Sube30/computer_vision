import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov9c.pt')

# Open the video file
video_path = "fall.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width,frame_height)
res = cv2.VideoWriter('fall_res.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         30, size) 
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break
    fall = 0
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, classes = 0,persist=True,conf=0.75, iou=0.2, tracker="botsort.yaml" )
        detections = []
        for result in results:
            for box in result.boxes:
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf  # Confidence score
                w = x2-x1
                h = y2 - y1
                aspect_ratio = w/h 
                if w/h > 1.4:
                    fall+=1
                    
                    cv2.putText(frame,"Fall Detected",(int(x1),int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        res.write(frame)
# Release the video capture object and close the display window
cap.release()
res.release()
