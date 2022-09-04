'''
For tasks 1 and 2:
I implemented a simple binary classification conv-net

architecture:
    3*[conv2d->max_pool]->MLP

This model is enough to overfit my cam dataset

'''
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from tensorflow import keras
from keras import Sequential
from keras.layers import (
    Flatten, Dense, Dropout, 
    Conv2D, Input, MaxPooling2D
    )


class MotorAIModel:
    def __init__(
        self, 
        input_size: Tuple[int, int], 
        lr: float = 5e-4, 
        filters: List[int] = [16, 32, 64], 
        ks: int = 3, 
        num_classes: int = 2, 
        dropout: float = 0.3
        ) -> None:

        self.input_size = input_size
        self.lr = lr
        self.filters = filters
        self.ks = ks
        self.num_classes = num_classes
        self.dropout = dropout
        self.opt = keras.optimizers.Adam(learning_rate=self.lr)

        self.model = self._create_model()
        self.model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'], 
            optimizer=self.opt
        )


    def _create_model(self) -> tf.keras.Sequential:
        ''' Simple CNN model to overfit for tasks 1 and 2 '''
        model = Sequential()  
        model.add(Input(
            shape=(self.input_size[0], self.input_size[1], 3))
            )
        for num_filters in self.filters:
            model.add(Conv2D(
                filters=num_filters, 
                kernel_size=self.ks, 
                activation='relu')
                )
            model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dropout(self.dropout))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model


    def __call__(self) -> tf.keras.Sequential:
        return self.model


    def __repr__(self) -> str:
        self.model(np.ones((1, self.input_size[0], self.input_size[1], 3)))
        self.model.summary() 
        return ''


if __name__ == '__main__':
    model = MotorAIModel(input_size=(256, 256))
    print(model)
