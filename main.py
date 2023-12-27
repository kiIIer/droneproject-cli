import argparse

import cv2
import numpy as np
import onnxruntime as ort
from keras import models

from object_analyze import analyze_image, draw_boxes
from presence_detector import detect_presence


def wait_for_presence(presence_model, frame, threshold):
    presence = detect_presence(presence_model, frame)
    print(presence)
    return presence >= threshold


def update_boxes(object_session, frame, object_threshold, iou_threshold):
    return analyze_image(object_session, frame, object_threshold, iou_threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Path to the video file', required=True)
    parser.add_argument('--presence-model-path', type=str, help='Path to the Keras model file', required=True)
    parser.add_argument('--object-model-path', type=str, help='Path to the ONNX model file', required=True)
    parser.add_argument('--threshold-presence', type=float, default=0.5, help='Threshold for presence detection')
    parser.add_argument('--threshold-object', type=float, default=0.5, help='Threshold for object detection confidence')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='Threshold for object detection IOU')
    parser.add_argument('--output', type=str, default='output.mp4', help='Directory to save detected frames')
    parser.add_argument('--play-video', action='store_true', help='Play the video after processing')
    parser.add_argument('--use-camera', action='store_true', help='Use camera instead of video file')
    parser.add_argument('--skip-frames', type=int, default=60, help='Number of frames to skip between detections')

    args = parser.parse_args()

    presence_model = models.load_model(args.presence_model_path)
    object_session = ort.InferenceSession(args.object_model_path)

    sus_detected = False
    boxes = np.array([])

    frame_n = 0

    if args.use_camera:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Error opening camera")
        frame_width = 640
        frame_height = 480
        fps = 30
    else:
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            raise IOError("Error reading video stream or file at " + args.video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

    if not args.play_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            error_msg = "Error reading frame from camera" if args.use_camera else "Error reading video stream or file at " + args.video_path
            raise IOError(error_msg)

        if frame_n % args.skip_frames == 0:
            if sus_detected:
                boxes = update_boxes(object_session, frame, args.threshold_object, args.iou_threshold)
            else:
                if wait_for_presence(presence_model, frame, args.threshold_presence):
                    sus_detected = True
                    continue

        boxed_frame = draw_boxes(frame, boxes)
        frame_n += 1
        if len(boxes) == 0:
            sus_detected = False
        if args.play_video:
            cv2.imshow('Video', boxed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            out.write(boxed_frame)

    cap.release()
    if args.play_video:
        cv2.destroyAllWindows()
    else:
        out.release()


if __name__ == '__main__':
    main()
