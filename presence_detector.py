import cv2
import numpy as np
from keras.preprocessing.image import img_to_array


def process_frame(frame, model, input_shape):
    frame_resized = cv2.resize(frame, (input_shape[1], input_shape[0]))

    if input_shape[2] == 1:
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frame_resized = frame_resized.reshape(input_shape)

    frame_normalized = img_to_array(frame_resized) / 255.0
    frame_normalized = np.expand_dims(frame_normalized, axis=0)

    prediction = model.predict(frame_normalized)
    return prediction[0][0]


def detect_presence(model, frame):
    input_shape = model.input_shape[1:4]
    presence_detected = process_frame(frame, model, input_shape)
    return presence_detected
