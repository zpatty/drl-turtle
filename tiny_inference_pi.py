import cv2
import time
import darknet

# Load the custom Tiny YOLO model with custom weights
config_file = "path/to/yolov4-tiny-turtle_1.cfg"
weights_file = "path/to/yolov4-tiny-turtle_5000.weights"
data_file = "path/to/turtle_yolov4_tiny.data"

network, class_names, class_colors = darknet.load_network(
    config_file, data_file, weights_file, batch_size=1
)

# Update class_names list with the single class name "turtle"
class_names = ['turtle']

#Image Version
#--------------------------
img = cv2.imread('test.jpg') # Test Image

# Convert image to Darknet format
darknet_image = darknet.make_image(img.shape[1], img.shape[0], 3)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
darknet.copy_image_from_bytes(darknet_image, img_rgb.tobytes())

# Run the inference
detections = darknet.detect_image(network, class_names, darknet_image)

# Release the Darknet image
darknet.free_image(darknet_image)

# Show results
for detection in detections:
    class_name, confidence, (x, y, w, h) = detection
    print("Rect:", class_name, confidence, (x, y, x+w, y+h))

# Release the network 
darknet.free_network_ptr(network)
#--------------------------
# #Video Version
# #--------------------------
# # Open video capture
# video_path = 'path/to/video.mp4'
# cap = cv2.VideoCapture(video_path) # Live inference on a video
# #cap = cv2.VideoCapture(0)  # Webcam

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to Darknet format
#     darknet_image = darknet.make_image(frame.shape[1], frame.shape[0], 3)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())

#     # Run the inference
#     detections = darknet.detect_image(network, class_names, darknet_image)

#     # Release the Darknet image
#     darknet.free_image(darknet_image)

#     # Draw bounding boxes and labels on the frame
#     for detection in detections:
#         class_name, confidence, (x, y, w, h) = detection
#         x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

#     # Display the frame
#     cv2.imshow("Tiny YOLO Live Inference", frame)

#     # Exit when 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release webcam capture and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

# # Release the network 
# darknet.free_network_ptr(network)
# #--------------------------