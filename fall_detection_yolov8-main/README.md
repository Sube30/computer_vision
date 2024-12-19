# fall_detection_yolov8

Follow the steps below to set up the environment and install dependencies.

**Clone this repository:**

git clone https://github.com/Sube30/fall_detection_yolov8.git

cd fall_detection_yolov8

**Create and activate a virtual environment (optional but recommended):**

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

**Install the required dependencies:**

pip install -r requirements.txt


**Usage**


To detect a fall in a image, run the following command:

python fall_v8_image.py 

To detect a fall in a video, run the following command:

python fall_v8_final.py 


# Results

Bounding box drawn around the person.

Keypoints for human pose detected and displayed as red dots.

Fall detection result based on the human pose.

Example:

Bounding box: (0.0, 0.0, 1515.65, 307.16)

Fall detected: True


# Acknowledgements
YOLOv8 for object detection and pose estimation.
OpenCV for image processing and visualization.
NumPy for numerical computing.
