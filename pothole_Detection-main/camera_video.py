#importing necessary libraries
import cv2 as cv
import time
import geocoder
import os

#reading label name from obj.names file
class_name = []
with open(os.path.join("project_files",'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

#importing model weights and config file
#defining the model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')

# ✅ FIX: Use CPU Instead of CUDA
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

#defining the video source (0 for camera or file name for video)
cap = cv.VideoCapture("test.mp4") 
width  = cap.get(3)
height = cap.get(4)
result = cv.VideoWriter('result.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10,(int(width),int(height)))

# ✅ FIX: Handle Geocoder API Failure
try:
    g = geocoder.ip('me')
except:
    g = None

#defining parameters for result saving and get coordinates
result_path = "pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = time.time()

#detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if ret == False:
        break

    # Detect potholes
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w*h
        area = width*height
        
        # Draw detection boxes
        if(len(scores)!=0 and scores[0]>=0.7):
            if((recarea/area)<=0.1 and box[1]<600):
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 1)
                cv.putText(frame, "%" + str(round(scores[0]*100,2)) + " " + label, (box[0], box[1]-10),cv.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 1)
                
                # Save image every 2 seconds
                if((time.time()-b)>=2):
                    cv.imwrite(os.path.join(result_path,'pothole'+str(i)+'.jpg'), frame)
                    with open(os.path.join(result_path,'pothole'+str(i)+'.txt'), 'w') as f:
                        if g is not None:
                            f.write(str(g.latlng))
                        else:
                            f.write("Location not available")
                    b = time.time()
                    i = i+1

    # Show FPS
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    cv.putText(frame, f'FPS: {fps}', (20, 50), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Display result
    cv.imshow('frame', frame)
    result.write(frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
result.release()
cv.destroyAllWindows()
os._exit(0)
