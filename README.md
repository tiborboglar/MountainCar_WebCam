# Control a car with your Web Cam! :D

* Fun repository with an end-to-end solution to control OpenAI's Mountain Car using your own webcam. The model is based on Tensorflow Keras.

`capture_dataset.py`: Use CLI to save frames from your webcam, you can define whether the frames splits are used for training/testing/validating, and also the label names.

`dataloader.py`: Easily generate a tensorflow data loader for each split (train/val/test), applying normalization and some baisc data augmentation. Contains visualization function to visualize a batch of data, and ensures augmentation only on training split.

`model.py`: Defines a simple Conv-Net `3*[conv2d->max_pool]->MLP`, with compilation on its constructor, making it easy to load weights after saving model.

`train.py`: Script that constructs a model and dataset, train the model, and dump its weights plus frame rate capability of the network. 

`control_car.py`: Open OpenAI gym environment, load webcam, load model, render environment and predicts the action based on the network output. If hand is closed, action moves car to the left, else it will move to the right.
