'''
Author: Tibor Camargo

Simple script to load dataset created from webcam
'''
from typing import Tuple, Union
import matplotlib.pyplot as plt
from keras.preprocessing.image import (
    ImageDataGenerator, DirectoryIterator
    )


class DataLoader:
    '''
    Create a simple data loader from webcam using the structure:

    |__ (dir) path/to/cam
    |_______ (dir) split/
    |_____________ (dir) open/
    |_____________ (dir) closed/

    with split being 'train', 'val', 'test'
    '''
    def __init__(
        self,
        train_root_dir: str,
        val_root_dir: Union[str, None], 
        test_root_dir: Union[str, None], 
        input_size: Tuple[int, int],
        batch_size: int
        ) -> None:

        self.input_size = input_size
        self.batch_size = batch_size
        self.train_root_dir = train_root_dir
        self.val_root_dir = val_root_dir
        self.test_root_dir = test_root_dir

        self.train = self.generator(split='train', folder=self.train_root_dir)
        if val_root_dir:
            self.val = self.generator(split='val', folder=self.val_root_dir)
        if test_root_dir:
            self.test = self.generator(split='test', folder=self.test_root_dir)


    def generator(self, split: str, folder: str) -> DirectoryIterator:
        if split == 'train':
            generator = ImageDataGenerator(rescale=1./255, rotation_range=15) 
        else:
            generator = ImageDataGenerator(rescale=1./255)
        
        dataset = generator.flow_from_directory(
            directory=self.train_root_dir,
            target_size=self.input_size,
            color_mode='rgb',
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=1
            )

        return dataset


    def visualize(self, split: str) -> None:
        """ Visualize a batch of your data split """
        if split == 'train':
            dataset = self.train
        if split == 'val':
            dataset = self.val
        if split == 'test':
            dataset = self.test

        for i, data in enumerate(dataset):
            x, y = data
            fig, ax = plt.subplots(1, x.shape[0])
            fig.set_size_inches((15, 15))
            for i in range(x.shape[0]):
                ax[i].imshow(x[i])
            plt.show()
            break
