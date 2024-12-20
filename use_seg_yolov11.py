from ultralytics import YOLO
import cv2
import random
import numpy as np

def overlay(image, mask, color, alpha=1, resize=None):
    """Combines image and its segmentation mask into a single image without transparency."""
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    # We use alpha=1 for no transparency (fully opaque)
    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

# Load a model
model = YOLO("yolo11x-seg.pt")
class_names = model.names
print('Class Names: ', class_names)

# Define colors for visualization
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# Open the video file
cap = cv2.VideoCapture('C://work//human_blur//testvedio03.mp4')

while True:
    success, img = cap.read()
    if not success:
        break

    h, w, _ = img.shape
    results = model.predict(img, stream=True)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            # Only process "person" class (class index 0)
            if int(box.cls) == 0:  # Class 0 is "person" in this case
                seg = cv2.resize(seg, (w, h))
                img = overlay(img, seg, colors[0], alpha=1)  # No transparency (alpha=1)

    # Display the image with "person" detection
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
