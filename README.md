# MotorAIChallenge

### Task 1:
* a) What gestures you used and why you chose those specific gestures

I chose two gestures, open and closed hands. I chose them because I think it is a pretty natural and intuitive binary system humans implement when gesturing, and also because it seems quite easy to distinguish an open hand and a closed hand. 

* b) Explain the design strategies you used to create the dataset

I simply created directories for `training` and `test`, where each directory comprises of two folders - `open` and `closed` - corresponding to images of its respective label. 

Actually the test dataset here is not useful, in the sense that my environment is super controlled and I just needed to overfit my own data. But I created nevertheless, just to ensure that the training accuracy was correct.  

* c) What risks are you aware of in your dataset: where it might fail and why?

My dataset was created under very specific conditions, such as: lighting, distance of my hand, inclination, background, etc. It might fail under any other conditions, for example if my light is way more lit than it was by the time I recorded it. 

The reason, as mentioned, is that I simply overfit my data, so I don't know which features were learned for determining if my hand was closed or open, it could be that the network learned that my hand is open because there is simply less skin-colored pixels in the scene, but I can only say that by assessing other data, and investigating which regions of an image is more activated by the network weights. 

* d) What were the difficulties you faced and how did you overcome them? (If there
were any difficulties)

No difficulties.

* e) Describe the properties and statistics about the dataset

Monocular RGB images using a lossless format (.png)
Training size: 2019 open hand images, 2022 closed hand images
Test size: 100 open hand images, 100 closed hand images
