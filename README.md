# Face detection with Convolution Neural Networks using PyTorch

## Dependencies
- Python 3.7
- opencv-python 4.8.1.78
- pandas 1.3.5
- torch 1.10.0
- torchsummary 1.5.1
- torchvision 0.11.1
- Run generateData.py to generate the dataset

## Approach
A functioning face detection system was created using the pre-trained Haar Cascade classifier in the previous task, however its accuracy is low and the detection is unreliable. As an alternative, task 2 involves of designing and implementing a custom face detection system from scratch using Convolutional Neural Networks. As part of the task specification, the dataset must also be collected and pre-processed for training.

To create a dataset that can be used to train the system, we can use the same images as the earlier task, however this time instead of having the guideline be the number of faces, we manually label the data with bounding boxes for the faces as described in the limitations section of the previous task. For this is used. To figure out the coordinates of the faces of each of the images, we implemented a mouse tracking script (see `classifyData.py`). The image can be loaded by changing the `image_path` variable and using the mouse, the coordinates of the two opposing corners of the square bounding each face can be found. We store these values in the groundTruth.json file in the testData folder. In this example test dataset there are 16 mapped images containing faces, and 10 without.

After the labelled data is created, to generate the data needed to train the CNN, generateData.py is used. A CNN required images of a pre-set dimension. The program takes each image in groundTruth.json in the testData folder, and strides a square window of 200x200 pixels, 20 pixels at a time, saving each `jpg` inside of the `trainingData` folder. With every iteration of the sub images `for`-loop, the program checks if there is an intersection between the 200x200 pixel sub-image and the bounding boxes from the `groundTruth.json` file. If there is an intersection, we make a note of the name of the sub-image file and the fact that there was a face in that sub image, and we create a new `groundTruth.json` file in trainingData folder containing this information. entries for sub-images with faces will have the "has_face" attribute set to 1.

Having succesfully gotten a dataset that can be used to train the CNN, the dataset needs to be converted into a format where the results can be used. For that we create a CSV file with the data points needed.

To prepare the dataset, a dataset class can be made by subclassing the pytorch dataset class. We load the labels from the CSV file and split the dataset into portions to be used for training and a smaller portion to be used to evaluate the set after for validation.

Since image transformations are an essential part of deep learning models, there's a section to resize, expand, or normalize it to give diversity in model performance. To avoid writing code that loops over datasets, we use dataloaders.

The bulk of the program needed to create a CNN is under defining binary classifiers. The network class defines the neural network class that is inherited from the nn.Module. It also consists of four convolutional layers, two fully connected layers and implements the forward pass of the network. 

To quantify the performance of the network and to compute gradients to update the network weights, PyTorch uses loss functions. For binary classification like this project, due to numerical stability and speed, log_softmax is the logical option.

Where the predicted and correct results differ, an optimiser is used to make calculate gradients with respect to the loss function and compute how to change the weights of the different nodes in the neural network to minimise the loss. 

We set aside a few images for evaluating the performance of the model and use the rest of the dataset training. At each training batch, we compute the training and test loss with respect to the training and test data sets respectively, and optimise the network till the test loss starts increasing (i.e., model starts overfitting)

## Limitations

Still under construction 
very very WIP

