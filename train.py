'''
Author: Tibor Camargo

Simple script to train a binary classification model
'''
import time
import numpy as np
from pathlib import Path
from model import MotorAIModel
from dataloader import DataLoader


CKPT_DIR = r'saved_models/my_model'
TRAIN_ROOT_DIR = r'data/cam/train'
TEST_ROOT_DIR = r'data/cam/test'
INPUT_SIZE = (256, 256)
BATCH_SIZE = 16
CNN_FILTERS = [16, 32, 64]


if __name__ == '__main__':
    # Retrieve data 
    dataloader = DataLoader(TRAIN_ROOT_DIR, None, TEST_ROOT_DIR, INPUT_SIZE, BATCH_SIZE)
    train_ds = dataloader.train

    # Create model 
    MotorModel = MotorAIModel(input_size=INPUT_SIZE, filters=CNN_FILTERS)
    model = MotorModel()

    # Train model and save to disk
    history = model.fit(x=train_ds, epochs=5, shuffle=True, verbose=1)

    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    model.save(CKPT_DIR)

    # Log frame rate using batch size
    total_time = 0
    num_runs = 50
    X = np.ones(shape=(BATCH_SIZE, INPUT_SIZE[0], INPUT_SIZE[1], 3))
    for i in range(num_runs):
        start = time.time()
        model(X)
        elapsed = time.time() - start
        total_time += elapsed
    fps = (num_runs*BATCH_SIZE)/total_time
    with open(r'fps.txt', 'w') as f:
        f.write(f'FPS: {str(round(fps, 2))}')
        