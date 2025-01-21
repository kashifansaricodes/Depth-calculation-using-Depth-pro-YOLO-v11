from PIL import Image
import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO

yolo_model = YOLO('yolo11s.pt')

image_path = "data/example.jpg"

yolo_input = cv2.imread(image_path)

results = yolo_model(yolo_input)

person_boxes = []
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, cls in zip(boxes, classes):
        if result.names[int(cls)] == "person":
            x1, y1, x2, y2 = map(int, box[:4])
            person_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(yolo_input, (x1, y1), (x2, y2), (0, 255, 0), 2)



# Resize the image to a desired width and height
desired_width = 600  # Adjust to fit your monitor
desired_height = 450  # Adjust to fit your monitor


# Load depth model and processing transformation
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

image, _, f_px = depth_pro.load_rgb(image_path)
depth_input = transform(image)
## Convert NumPy array to PyTorch tensor
# depth_input = torch.from_numpy(depth_input).unsqueeze(0)  # Add batch dimension

# # Move tensor to the same device as the model
# device = next(depth_model.parameters()).device
# depth_input = depth_input.to(device)

prediction = depth_model.infer(depth_input, f_px=f_px)
depth = prediction["depth"]  # Depth in meters

depth_np = depth.squeeze().cpu().numpy()

for x1, y1, x2, y2 in person_boxes:
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    depth_value = depth_np[center_y, center_x]

    text = f'Depth: {depth_value:.2f} m'
    front = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 3
    text_size = cv2.getTextSize(text, front, font_scale, font_thickness)[0]

    text_x = x1
    text_y = y1 - 10
    rect_x1 = text_x - 2
    rect_y1 = text_y - text_size[1] - 4  # Fixed variable name from rect_x1 to rect_y1
    rect_x2 = text_x + text_size[0] + 2
    rect_y2 = text_y + 2

    cv2.rectangle(yolo_input, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
    cv2.putText(yolo_input, text, (text_x, text_y), front, font_scale, (255, 255, 255), font_thickness)


yolo_input_resized = cv2.resize(yolo_input, (desired_width, desired_height))

cv2.imshow('Person Detection with Depth', yolo_input_resized)
cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('person_detection_with_depth.jpg', yolo_input_resized)

depth_np_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
inv_depth_np_normalized = 1.0 - depth_np_normalized  # Invert the normalized depth values
depth_colormap = cv2.applyColorMap((inv_depth_np_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

depth_colormap_resized = cv2.resize(depth_colormap, (desired_width, desired_height))

cv2.imshow('Inverted Depth Colormap', depth_colormap_resized)
cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('inverted_depth_colormap.jpg', depth_colormap_resized)