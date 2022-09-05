'''
Author: Tibor Camargo

Simple script to create dataset for a binary classification problem
'''
import os
import cv2
import argparse
import logging
from pathlib import Path


parser = argparse.ArgumentParser(
    description='Create your custom dataset using webcam, overwrites existing data'
    )

parser.add_argument(
    '-l', '--label', 
    help='Capture closed/open hand dataset', 
    type=str,
    required=True, 
    choices=['open', 'closed']
    )

parser.add_argument(
    '-s', '--split', 
    help='Dataset used for train/val/test', 
    type=str,
    required=True, 
    choices=['train', 'val', 'test']
    )


if __name__ == '__main__':
    # Retrieve arguments from CLI
    args = vars(parser.parse_args())

    # Create folders to save dataset
    hand_dir = os.path.join('data', 'cam', args['split'], args['label'])
    Path(hand_dir).mkdir(parents=True, exist_ok=True)
    logging.info(f'Saving files to: "{os.path.abspath(hand_dir)}"')

    # Starting video recording
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        ret, frame = cap.read()

        # Ignore initial frame because it is blank, resize frame 
        if i > 0:
            frame = cv2.resize(frame, dsize=(256, 256))
            dst = os.path.join(hand_dir, '{:04d}.png'.format(i))
            cv2.imshow('frame', frame)
            cv2.imwrite(dst, frame)
            cv2.waitKey(1)
        
        # Display info to keep track of number of files saved to disk
        if i%100 == 0:
            logging.info(f'Saving frame: {i}')
        i+=1
        
