def setup_cli(subparsers):
    parser = subparsers.add_parser('detect_object', help='Detect objects in a video using an ONNX model')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model file')
    parser.add_argument('--output-dir', type=str, default='output_objects',
                        help='Directory to save frames with detected objects')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for object detection confidence')
    parser.set_defaults(func=detect_object)


import os

import cv2


def detect_object(args):
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load the ONNX model
    net = cv2.dnn.readNetFromONNX(args.model_path)

    # Initialize video reader and writer
    cap = cv2.VideoCapture(args.video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = os.path.join(args.output_dir, 'output.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and prepare the frame for model input
        resized_frame = cv2.resize(frame, (640, 640))
        blob = cv2.dnn.blobFromImage(resized_frame, 1.0, (640, 640), [104, 117, 123], swapRB=True)
        net.setInput(blob)

        # Run inference
        detections = net.forward()

        # Process each detection
        for detection in detections[0]:
            confidence = detection[4]
            if confidence > args.threshold:
                class_id = int(detection[5])
                x, y, w, h = detection[0:4]

                # Rescale coordinates
                x1 = int((x - w / 2) * frame_width)
                y1 = int((y - h / 2) * frame_height)
                x2 = int((x + w / 2) * frame_width)
                y2 = int((y + h / 2) * frame_height)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved to {output_path}")
