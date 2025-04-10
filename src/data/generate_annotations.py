import os
import cv2
import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm

def generate_annotations_for_dataset(dataset_root, languages):
    """Generate bounding box annotations for all images in the dataset."""
    
    # Load a pre-trained text detection model (EAST)
    print("Loading text detection model...")
    
    # Path to the pre-trained EAST text detector
    east_model_path = os.path.join('models', 'frozen_east_text_detection.pb')
    
    # Download the model if it doesn't exist
    if not os.path.exists(east_model_path):
        os.makedirs('models', exist_ok=True)
        print("Downloading EAST text detection model...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb",
            east_model_path
        )
    
    net = cv2.dnn.readNet(east_model_path)
    
    # Check for MPS GPU availability (Apple Silicon)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS GPU is available. Using GPU acceleration.")
        # OpenCV doesn't directly support MPS, but can use OpenCL which may leverage GPU
        # Try to set DNN backend to use GPU acceleration if available
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
            print("Using OpenCL acceleration for OpenCV DNN.")
        except:
            print("OpenCL acceleration failed, falling back to CPU.")
    else:
        print("MPS GPU not available. Using CPU.")
    
    # Process each language folder
    for lang in languages:
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(dataset_root, lang, split)
            images_dir = os.path.join(split_dir, 'images')
            
            if not os.path.exists(images_dir):
                print(f"Directory not found: {images_dir}")
                continue
                
            # Create annotations directory if it doesn't exist
            annotations_dir = os.path.join(split_dir, 'annotations')
            os.makedirs(annotations_dir, exist_ok=True)
            
            # Read image paths
            with open(os.path.join(split_dir, 'images.txt'), 'r') as f:
                image_files = f.read().splitlines()
            
            # Read corresponding labels
            with open(os.path.join(split_dir, 'labels.txt'), 'r') as f:
                labels = f.read().splitlines()
            
            print(f"Processing {lang}/{split}: {len(image_files)} images")
            
            # Process each image
            for img_file, text_label in tqdm(zip(image_files, labels), total=len(image_files)):
                img_path = os.path.join(images_dir, os.path.basename(img_file))
                
                # Skip if image doesn't exist
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue
                
                # Read the image
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                
                orig_h, orig_w = image.shape[:2]
                
                # Prepare the image for text detection
                blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320), 
                                            (123.68, 116.78, 103.94), swapRB=True, crop=False)
                
                # Set the input to the network
                net.setInput(blob)
                
                # Get output layers
                output_layer_names = [
                    "feature_fusion/Conv_7/Sigmoid",
                    "feature_fusion/concat_3"
                ]
                
                # Forward pass to get scores and geometry
                (scores, geometry) = net.forward(output_layer_names)
                
                # Decode predictions
                (rects, confidences) = decode_predictions(scores, geometry, min_confidence=0.5)
                
                # Apply non-maximum suppression
                boxes = non_max_suppression(np.array(rects), probs=confidences)
                
                # If no text is detected, use the whole image as a bounding box
                if len(boxes) == 0:
                    boxes = np.array([[0, 0, orig_w, orig_h]])
                
                # Scale the bounding boxes back to the original image size
                ratio_h = orig_h / 320
                ratio_w = orig_w / 320
                
                # Convert to XYWH format (we'll store as oriented bounding boxes)
                annotations = []
                for (startX, startY, endX, endY) in boxes:
                    startX = int(startX * ratio_w)
                    startY = int(startY * ratio_h)
                    endX = int(endX * ratio_w)
                    endY = int(endY * ratio_h)
                    
                    # Create oriented bounding box (4 points)
                    # For now, we'll use axis-aligned boxes
                    box = [
                        [startX, startY],  # top-left
                        [endX, startY],    # top-right
                        [endX, endY],      # bottom-right
                        [startX, endY]     # bottom-left
                    ]
                    
                    annotations.append({
                        'points': box,
                        'text': text_label,
                        'language': lang
                    })
                
                # Save annotations to a JSON file
                annotation_file = os.path.join(annotations_dir, os.path.splitext(os.path.basename(img_file))[0] + '.json')
                with open(annotation_file, 'w') as f:
                    json.dump(annotations, f, indent=2)

def decode_predictions(scores, geometry, min_confidence):
    """Decode text detection predictions."""
    (num_rows, num_cols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, num_rows):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        for x in range(0, num_cols):
            if scores_data[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = angles_data[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x_data0[x] + x_data2[x]
            w = x_data1[x] + x_data3[x]

            endX = int(offsetX + (cos * x_data1[x]) + (sin * x_data2[x]))
            endY = int(offsetY - (sin * x_data1[x]) + (cos * x_data2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scores_data[x])

    return (rects, confidences)

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    """Apply non-maximum suppression to avoid detecting the same text region multiple times."""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

if __name__ == "__main__":
    dataset_root = 'Dataset'
    languages = ['bengali', 'hindi', 'kannada', 'tamil', 'telugu']
    generate_annotations_for_dataset(dataset_root, languages)
