import depth_pro
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

# Path to the input image
image_path = "data/kashif.jpg"

# Load YOLO model
yolo_model = YOLO('yolo11s.pt')

# Read the input image using OpenCV
yolo_input = cv2.imread(image_path)

# Perform object detection using YOLO model
results = yolo_model(yolo_input)

# Initialize list to store bounding boxes of detected persons
person_boxes = []
for result in results:
    # Get bounding boxes and class labels from the result
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # Iterate through each detected object
    for box, cls in zip(boxes, classes):
        # Check if the detected object is a person
        if result.names[int(cls)] == "person":
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            person_boxes.append((x1, y1, x2, y2))
            # Draw bounding box on the image
            cv2.rectangle(yolo_input, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Resize the image to a desired width and height
desired_width = 450  # Adjust to fit your monitor
desired_height = 600  # Adjust to fit your monitor

# Load depth model and processing transformation
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval() # Set the model to evaluation mode

# Load and transform the image for depth estimation
image, _, f_px = depth_pro.load_rgb(image_path) # Load RGB image and focal length
depth_input = transform(image) # Transform the image for depth estimation

# Perform depth estimation
prediction = depth_model.infer(depth_input, f_px=f_px) # Load the depth model and perform inference
depth = prediction["depth"]  # Depth in meters 

# Convert depth tensor to numpy array
depth_np = depth.squeeze().cpu().numpy()

# Iterate through each detected person bounding box
for x1, y1, x2, y2 in person_boxes:
    # Calculate the center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Get the depth value at the center of the bounding box
    depth_value = depth_np[center_y, center_x]

    # Prepare text to display depth value
    text = f'Depth: {depth_value:.2f} m'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Calculate text position and background rectangle coordinates
    text_x = x1
    text_y = y1 - 10
    rect_x1 = text_x - 2
    rect_y1 = text_y - text_size[1] - 4  # Fixed variable name from rect_x1 to rect_y1
    rect_x2 = text_x + text_size[0] + 2
    rect_y2 = text_y + 2

    # Draw background rectangle for text
    cv2.rectangle(yolo_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    # Draw text on the image
    cv2.putText(yolo_input, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# Resize the image with bounding boxes and depth values
yolo_input_resized = cv2.resize(yolo_input, (desired_width, desired_height))

# Display the image with person detection and depth values
cv2.imshow('Person Detection with Depth | press any key to continue', yolo_input_resized)
cv2.waitKey(0)

# Save the image with person detection and depth values
cv2.imwrite('person_detection_with_depth.jpg | press any key to continue', yolo_input_resized)

# Normalize the depth values for visualization
depth_np_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
# Invert the normalized depth values
inv_depth_np_normalized = 1.0 - depth_np_normalized
# Apply color map to the inverted depth values
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

# Resize the depth colormap image
depth_colormap_resized = cv2.resize(depth_colormap, (desired_width, desired_height))

# Display the inverted depth colormap
cv2.imshow('Inverted Depth Colormap | press any key to continue', depth_colormap_resized)
cv2.waitKey(0)
# Save the inverted depth colormap image
cv2.imwrite('inverted_depth_colormap.jpg', depth_colormap_resized)