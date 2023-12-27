import cv2


def play_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video or error in reading video.")
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Example usage
video_path = 'test-files/3_boxed.mp4'  # Replace with your video file path
play_video(video_path)
