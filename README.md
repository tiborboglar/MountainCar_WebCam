# MotorAIChallenge

### Task 1:
* a) What gestures you used and why you chose those specific gestures

I chose two gestures, open and closed hands. I chose them because I think it is a pretty natural and intuitive binary system humans implement when gesturing, and also because it seems quite easy to distinguish an open hand and a closed hand. 

* b) Explain the design strategies you used to create the dataset

I simply created directories for `training` and `test`, where each directory comprises of two folders - `open` and `closed` - corresponding to images of its respective label. 

The percentage of images in each label/class was kept in almost 50/50 for simplicity, and I decided to use small images to make it easy to overfit and for fast training and inferences.

Actually the test dataset here is not useful, in the sense that my environment is super controlled and I just needed to overfit my own data. But I created nevertheless, just to ensure that the training accuracy was correct.  


* c) What risks are you aware of in your dataset: where it might fail and why?

My dataset was created under very specific conditions, such as: lighting, distance of my hand, inclination, background, etc. It might fail under any other conditions, for example if my light is way more lit than it was by the time I recorded it. 

The reason, as mentioned, is that I simply overfittted my data, so I don't know which features were learned for determining if my hand was closed or open, it could be that the network learned that my hand is open because there is simply less skin-colored pixels in the scene, but I can only say that by assessing other data, and investigating which regions of an image is more activated by the network weights. 

* d) What were the difficulties you faced and how did you overcome them? (If there
were any difficulties)

No difficulties.

* e) Describe the properties and statistics about the dataset

Monocular RGB images of size 256x256 using a lossless format (.png)

Training size: 2019 open hand images, 2022 closed hand images

Test size: 100 open hand images, 100 closed hand images


### Task 2:

* a) Evaluation strategies/metrics used for both the model and the task.

Metrics: Accuracy, since it is a binary problem with a class balance of 50/50, otherwise I would need to think about false negatives, false positives, and so on and so forth.

* b) Explain your code flow and the results within the report.

`capture_dataset.py`: Use CLI to save frames from your webcam, you can define the split name and the label name.

`dataloader.py`: Easily generate a tensorflow data loader for each split (train/val/test), applying normalization and some baisc data augmentation. Contains visualization function to visualize a batch of data, and ensures augmentation only on training split.

`model.py`: Defines a simple Conv-Net `3*[conv2d->max_pool]->MLP`, with compilation on its constructor, making it easy to load weights after saving model.

`train.py`: Script that constructs a model and dataset, train the model, and dump its weights plus frame rate capability of the network. 

`control_car.py`: Open OpenAI gym environment, load webcam, load model, render environment and predicts the action based on the network output. If hand is closed, action moves car to the left, else it will move to the right.

* c) Evaluate your code in real-time and explain if your model works or not. If it works why it works, if it does not then why it doesn’t.

Qualitatively it works because I showed on video. Quantitatively the model outputs around 300 FPS on a GeForce GTX 1060 6gb, since the camera FPS is lower than 30, the bottleneck for real-time inference is the camera itself, since the runtime for rendering the environment and resizing images is almost negligible, but around 30 FPS is already easy to solve the problem smoothly.

* d) Explain within the report, your failed approaches in brief, what you learned from them

Fortunately my code worked instantly.

### Dataset Sample

![open](https://user-images.githubusercontent.com/25236592/188477310-c0c3d548-95ad-481c-8de5-014865400c4d.PNG)
![closed](https://user-images.githubusercontent.com/25236592/188477322-e54d1c3e-d1d1-4a75-bb13-057bb663ade8.PNG)

