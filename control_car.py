'''
Author: Tibor Camargo

Control MountainCar-v0 using Conv-Net trained on WebCam
'''
import cv2
import time
import numpy as np
from model import ControlCarModel
import gym


INPUT_SIZE = (256, 256)
CKPT_DIR = 'saved_models/my_model/'


if __name__ == '__main__':
    # Load model and weights
    Model = ControlCarModel(input_size=INPUT_SIZE)
    model = Model.model
    model.load_weights(CKPT_DIR)

    # Create openAI env
    env = gym.make('MountainCar-v0', new_step_api=True, render_mode='human')
    env.action_space.seed(42)
    observation, info = env.reset(seed=1, return_info=True)

    # Retrieve camera
    cap = cv2.VideoCapture(0)

    while True:
        # Read image from cam and normalize
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=INPUT_SIZE)
        frame = frame/255.

        # Apply prediction
        y_hat = model(frame[np.newaxis, :, :])
        label = np.argmax(y_hat)

        # Add prediction label to frame
        if label == 1:
            txt = 'open'
        else:
            txt = 'closed'

        frame = cv2.putText(
          img = frame,
          text = txt,
          org = (30, 30),
          fontFace = cv2.FONT_HERSHEY_DUPLEX,
          fontScale = 1.0,
          color = (0, 255, 0),
          thickness = 2
        )

        # Display cam
        cv2.imshow("frame", frame)
        cv2.waitKey(1)

        # Start gym and use label as action
        env.render()
        action = label

        if label == 1: 
            action += 1 # labels are 0 and 1, but 0 and 2 are needed
        observation, reward, done, truncated, info = env.step(action)

        # Reset animation if goal is reached
        if done:
            observation, info = env.reset(return_info=True)
