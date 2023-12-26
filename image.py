import cv2
import numpy as np
import onnxruntime as ort
import os


def load_and_preprocess_image(image, input_shape=(640, 640)):
    image_resized = cv2.resize(image, input_shape)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0

    image_transposed = np.transpose(image_normalized, (2, 0, 1))

    image_batch = np.expand_dims(image_transposed, axis=0)

    return image_batch, image_resized


def analyze_and_modify_image(onnx_model_path, image, threshold=0.5, iou_threshold=0.5):
    preprocessed_image, resized_image = preprocess_image(image)

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed_image})

    output_data = outputs[0][0]

    # Collect boxes that meet the threshold
    boxes = []
    scores = []
    class_confidences = []
    for detection in output_data:
        center_x, center_y, width, height, objectness, human_confidence, vehicle_confidence = detection
        if objectness > threshold:
            boxes.append([center_x, center_y, width, height])
            scores.append(objectness)
            class_confidences.append((human_confidence, vehicle_confidence))

    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Apply Non-Maximum Suppression
    selected_indices = non_max_suppression(boxes, scores, iou_threshold)
    selected_boxes = boxes[selected_indices]
    selected_class_confidences = np.array(class_confidences)[selected_indices]

    # Draw the selected boxes and class text on the image
    for box, class_confidence in zip(selected_boxes, selected_class_confidences):
        center_x, center_y, width, height = box
        scale_x, scale_y = resized_image.shape[1] / 640, resized_image.shape[0] / 640
        top_left_x = int((center_x - (width / 2)) * scale_x)
        top_left_y = int((center_y - (height / 2)) * scale_y)
        bottom_right_x = int((center_x + (width / 2)) * scale_x)
        bottom_right_y = int((center_y + (height / 2)) * scale_y)

        # Determine the class based on the higher confidence score
        class_text = "Human" if class_confidence[0] > class_confidence[1] else "Vehicle"

        # Draw the bounding box on the image
        cv2.rectangle(resized_image, (top_left_x, top_left_y),
                      (bottom_right_x, bottom_right_y),
                      (0, 255, 0), 2)

        # Put the class text below the bounding box
        text_location = (top_left_x, bottom_right_y + 15)  # Adjust the 15 pixels offset as needed
        cv2.putText(resized_image, class_text, text_location, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    return resized_image


def non_max_suppression(boxes, scores, iou_threshold):
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def preprocess_image(image, input_shape=(640, 640)):
    # Resize the image to the input shape
    image_resized = cv2.resize(image, input_shape)
    # Convert BGR (OpenCV default) to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    # Normalize the image
    image_normalized = image_rgb.astype(np.float32) / 255.0
    # HWC to CHW format
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    # Add a batch dimension
    image_batch = np.expand_dims(image_transposed, axis=0)
    return image_batch, image_resized


def process_video_to_frames(input_video_path, output_folder, onnx_model_path, threshold=0.5, iou_threshold=0.5):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError("Error opening video stream or file at " + input_video_path)

    frame_count = 0
    max_frames = 500  # Set a limit to the number of frames to process

    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            modified_frame = analyze_and_modify_image(onnx_model_path, frame, threshold, iou_threshold)

            # Save the modified frame to the output folder
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, modified_frame)

            frame_count += 1
    except Exception as e:
        print(f"Error processing video: {e}")
    finally:
        cap.release()

    print("Video processing completed. Processed frames:", frame_count)


# Example usage:
input_video_path = 'test-files/3.MP4'
output_folder = 'test-files/out'
onnx_model_path = 'test-files/best.onnx'

# Process the video
process_video_to_frames(input_video_path, output_folder, onnx_model_path)
