import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


def setup_cli(subparsers):
    parser = subparsers.add_parser('detect_presence', help='Detect presence in a video')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('model_path', type=str, help='Path to the Keras model file')
    parser.add_argument('--output-dir', type=str, default='output_presence', help='Directory to save detected frames')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for presence detection')
    parser.set_defaults(func=detect_presence)


def process_frame(frame, model, input_shape):
    # Resize the frame to match the model's expected input size
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[0]))

    # Convert the frame to grayscale if the model expects grayscale input
    if input_shape[2] == 1:  # Checking if the model expects 1 channel
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_resized = frame_resized.reshape(input_shape)  # Reshape to add channel dimension

    # Normalize the frame
    frame_normalized = img_to_array(frame_resized) / 255.0
    frame_normalized = np.expand_dims(frame_normalized, axis=0)

    prediction = model.predict(frame_normalized)
    return prediction[0][0]


def detect_presence(args):
    model = load_model(args.model_path)
    input_shape = model.input_shape[1:4]  # Adjust to get the correct shape (width, height, channels)
    cap = cv2.VideoCapture(args.video_path)
    os.makedirs(args.output_dir, exist_ok=True)
    detected_dir = os.path.join(args.output_dir, "detected")
    os.makedirs(detected_dir, exist_ok=True)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the prediction for the frame
        presence_detected = process_frame(frame, model, input_shape)

        # Check if the prediction is above the threshold
        if presence_detected >= args.threshold:
            frame_filename = os.path.join(detected_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Detected presence in frame {frame_count}, saved to {frame_filename}")

        frame_count += 1

    cap.release()
    print("Presence detection completed.")
