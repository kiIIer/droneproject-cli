import cv2
import numpy as np
import onnxruntime as ort


def load_and_preprocess_image(image, input_shape=(640, 640)):
    image_resized = cv2.resize(image, input_shape)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0

    image_transposed = np.transpose(image_normalized, (2, 0, 1))

    image_batch = np.expand_dims(image_transposed, axis=0)

    return image_batch, image_resized


def analyze_and_modify_image(onnx_model_path, image, threshold=0.5, iou_threshold=0.5):
    preprocessed_image, resized_image = load_and_preprocess_image(image)

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed_image})

    output_data = outputs[0][0]

    boxes = []
    scores = []
    for detection in output_data:
        center_x, center_y, width, height, objectness, human, vehicle = detection
        if objectness > threshold:
            boxes.append([center_x, center_y, width, height])
            scores.append(objectness)

    boxes = np.array(boxes)
    scores = np.array(scores)

    selected_indices = non_max_suppression(boxes, scores, iou_threshold)
    selected_boxes = boxes[selected_indices]

    for box in selected_boxes:
        center_x, center_y, width, height = box
        scale_x, scale_y = resized_image.shape[1] / 640, resized_image.shape[0] / 640
        top_left_x = int((center_x - (width / 2)) * scale_x)
        top_left_y = int((center_y - (height / 2)) * scale_y)
        bottom_right_x = int((center_x + (width / 2)) * scale_x)
        bottom_right_y = int((center_y + (height / 2)) * scale_y)

        cv2.rectangle(resized_image, (top_left_x, top_left_y),
                      (bottom_right_x, bottom_right_y),
                      (0, 255, 0), 2)

    cv2.imwrite('out.png', resized_image)


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


onnx_model_path = 'test-files/best.onnx'
image_path = 'test-files/Screenshot 2023-12-26 at 16.44.55.png'

analyze_and_modify_image(onnx_model_path, image_path, threshold=0.5)
