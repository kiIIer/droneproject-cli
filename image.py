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
    preprocessed_image, _ = preprocess_image(image)

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed_image})

    output_data = outputs[0][0]

    boxes = []
    objectness_scores = []
    for detection in output_data:
        center_x, center_y, width, height, objectness, _, _ = detection
        if objectness > threshold:
            boxes.append([center_x, center_y, width, height])
            objectness_scores.append(objectness)

    selected_indices = non_max_suppression(np.array(boxes), np.array(objectness_scores), iou_threshold)
    selected_boxes = [boxes[i] for i in selected_indices]

    return selected_boxes


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


def draw_rectangles_on_frame(frame, boxes, input_shape=(640, 640)):
    frame_height, frame_width = frame.shape[:2]
    scale_x, scale_y = frame_width / input_shape[0], frame_height / input_shape[1]

    for box in boxes:
        center_x, center_y, width, height = box
        center_x *= scale_x
        center_y *= scale_y
        width *= scale_x
        height *= scale_y

        top_left_x = int(center_x - width / 2)
        top_left_y = int(center_y - height / 2)
        bottom_right_x = int(center_x + width / 2)
        bottom_right_y = int(center_y + height / 2)

        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    return frame


def process_video(input_video_path, output_video_path, onnx_model_path, threshold=0.5, iou_threshold=0.5):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError("Error opening video stream or file at " + input_video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes = analyze_and_modify_image(onnx_model_path, frame, threshold, iou_threshold)
        modified_frame = draw_rectangles_on_frame(frame, boxes, input_shape=(640, 640))

        out.write(modified_frame)

    cap.release()
    out.release()


# Example usage:
input_video_path = 'test-files/3.MP4'
output_folder = 'test-files/3_boxed.mp4'
onnx_model_path = 'test-files/best.onnx'

# Process the video
process_video(input_video_path, output_folder, onnx_model_path)
