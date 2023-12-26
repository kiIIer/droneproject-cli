import argparse
import os
import cv2


def setup_cli(subparsers):
    parser = subparsers.add_parser('split', help='Split a video into frames')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--skip-frames', type=int, default=1, help='Number of frames to skip between saves')
    parser.add_argument('--output-dir', type=str, default='output_frames', help='Directory to save the output frames')
    parser.set_defaults(func=split_video)


def split_video(args):
    video_path = args.video_path
    skip_frames = args.skip_frames
    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            frame_filename = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f'Saved: {frame_filename}')

        frame_count += 1

    cap.release()
    print('Video splitting completed.')
