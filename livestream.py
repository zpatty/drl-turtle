import cv2
import numpy as np

# Load YOLO model with pre-trained weights
net = cv2.dnn.readNet("/Users/lisa/Desktop/turtle/trained_model_info/yolov4-tiny-turtle_1.cfg", "/Users/lisa/Desktop/turtle/trained_model_info/yolov4-tiny-turtle_5000.weights")

# Load COCO labels for Tiny YOLO
with open("/Users/lisa/Desktop/turtle/trained_model_info/turtle.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names (specific to YOLO Tiny)
output_layers = net.getUnconnectedOutLayersNames()

# Function to perform object detection on an image
def detect_objects(image):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image



# Function to play video and perform object detection in real-time
def play_video_with_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame
        frame = detect_objects(frame)

        # Display the frame
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/lisa/Desktop/turtle/live_detection/live_stream_test_video_low_qual.mov"  # Replace with your video file path
    play_video_with_detection(video_path)
