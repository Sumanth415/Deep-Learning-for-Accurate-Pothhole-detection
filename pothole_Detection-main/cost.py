# Importing necessary libraries
import cv2 as cv
import time
import geocoder
import os

# Read label names from obj.names file
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Load YOLOv4-tiny model 
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Video input and output settings
cap = cv.VideoCapture("test.mp4")
width = cap.get(3)
height = cap.get(4)
result = cv.VideoWriter('result.avi',
                        cv.VideoWriter_fourcc(*'MJPG'),
                        10, (int(width), int(height)))

# Handle Geocoder API failure
try:
    g = geocoder.ip('me')
except:
    g = None

# Constants for dimension an d cost estimation 
PIXELS_PER_INCH = 12.5  # Adjust based on calibration
DEPTH_INCHES = 3
COST_PER_CUBIC_FEET = 1000 

# Prepare output directory
result_path = "pothole_coordinates"
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Variables for tracking
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = time.time()
total_cost = 0  # Track total repair cost

# Detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        break

    # Detect potholes
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height

        # Filter valid potholes
        if len(scores) != 0 and scores[0] >= 0.7:
            if (recarea / area) <= 0.1 and box[1] < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(frame, "%" + str(round(scores[0] * 100, 2)) + " " + label, (x, y - 10),
                           cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

                # Save info every 2 seconds
                if (time.time() - b) >= 2:
                    img_path = os.path.join(result_path, f'pothole{i}.jpg')
                    txt_path = os.path.join(result_path, f'pothole{i}_details.txt')
                    cv.imwrite(img_path, frame)

                    # Estimate dimensions
                    length_inch = w / PIXELS_PER_INCH
                    breadth_inch = h / PIXELS_PER_INCH
                    volume_inch3 = length_inch * breadth_inch * DEPTH_INCHES
                    volume_ft3 = volume_inch3 / 1728
                    repair_cost = volume_ft3 * COST_PER_CUBIC_FEET
                    total_cost += repair_cost

                    # Save info
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(f"Location: {g.latlng if g else 'Not available'}\n")
                        f.write(f"Length: {length_inch:.2f} inches\n")
                        f.write(f"Breadth: {breadth_inch:.2f} inches\n")
                        f.write(f"Estimated Volume: {volume_ft3:.2f} ft³\n")
                        f.write(f"Estimated Repair Cost: INR {repair_cost:.2f}\n")

                    b = time.time()
                    i += 1

    # Display FPS
    ending_time = time.time() - starting_time
    fps = frame_counter / ending_time
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Show and save video
    cv.imshow('frame', frame)
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
result.release()
cv.destroyAllWindows()

# Save total cost
with open(os.path.join(result_path, 'total_cost.txt'), 'w', encoding='utf-8') as f:
    f.write(f"Total Estimated Repair Cost: INR {total_cost:.2f}\n")

os._exit(0)
