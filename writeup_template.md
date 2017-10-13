#**Behavioral Cloning** 

##**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[center_lane_driving]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[recover_1]: ./examples/recover_1.jpg
[recover_2]: ./examples/recover_2.jpg
[recover_3]: ./examples/recover_3.jpg
[original]: ./examples/original.jpg
[flipped]: ./examples/flipped.jpg

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
**model.py:** containing the script to create and train the model, i add some command line arguments(with default values) that makes possiable to speicify training hyper-parameters without changing code.
**drive.py:** for driving the car in autonomous mode, i made no change to this file.
**model.h5:** containing a trained convolution neural network 
**writeup_report.md:**  summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

Here is a brief description for classes and functions inside **model.py**:

*[table 1]*

|name | description|
|---- | ---|
|class **TrainConfig** | a simple class wrapper all the training configure as properties.|
|fuction **create_default_trainConfig**| create and return an instance of class TrainConfig containg default training configures.|
|function **load_training_info**|load the driving_log.csv file and parse the image names and steering angle|
|function **batch_generator**|a python generator function providing the batch data for feeding the model|
|function **trainModel**|define and training the convolutional model using keras|
|function **parse_args**|parse command line arguments|
|function **create_train_config_from_args**|create an instance of class TrainConfig from command line arguments|
|function **main**|the main function|

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), i add normalization and cropping operations to preprocess the input images, and i also add dropout to prevent overfitting.
Here is the model architecture using keras layers:

*[table 2]*

| Layer         |     Description|
|:---------------------:|:---------------------------------------------:|
|Lamda |normalize the input images, the input to this layer is 160x320x3 RGB images|
|Cropping2D |crop top and bottom portion with 75 and 20 pixels, keep only the section with road.|
|Conv2D|convolutional layer with 5x5 kernel size and 2x2 strides, 24 output channels, activate with 'relu' function|
|Dropout|dropout with 0.2 as the dropout probablity|
|Conv2D|convolutional layer with 5x5 kernel size and 2x2 strides, 36 output channels, activate with 'relu' function|
|Dropout|dropout with 0.2 as the dropout probablity|
|Conv2D|convolutional layer with 5x5 kernel size and 2x2 strides, 48 output channels, activate with 'relu' function|
|Dropout|dropout with 0.2 as the dropout probablity|
|Conv2D|convolutional layer with 5x5 kernel size and 1x1 strides, 64 output channels, activate with 'relu' function|
|Dropout|dropout with 0.2 as the dropout probablity|
|Conv2D|convolutional layer with 5x5 kernel size and 1x1 strides, 64 output channels, activate with 'relu' function|
|Dropout|dropout with 0.2 as the dropout probablity|
|Flatten|flatten the output of last convolutional layer to a single vector|
|Dense|fully connected layer using 'relu' activation function with 100 output units|
|Dropout|dropout with 0.2 as the dropout probablity|
|Dense|fully connected layer using 'relu' activation function with 50 output units|
|Dropout|dropout with 0.2 as the dropout probablity|
|Dense|fully connected layer using 'relu' activation function with 10 output units|
|Dropout|dropout with 0.2 as the dropout probablity|
|Dense|fully connected layer without activation function with 1 output unit as the prediction for steering angle|

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting, please refer to the above table 2.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, clockwise and anticlockwise driving, and collected more data at some places hard for driving, i also collected some data around track 2.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to choose a base model and experiment with it, made changes to the model and hyper-parameters according to the training and validataion loss, and use the trained model to test the auto-drive mode and found out where it failed and collect more data there or made some other changes to the model and hyper-parameters.

My first step was to use a convolution neural network model similar to the [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), I thought this model might be appropriate because it's designed by some professional self-driving experts and well tested.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added a dropout layer for each convolutional and fully connnected layer, which fix the overfitting problem.

Then I tuned the hyper-parameters such as batch size, epochs and dropout probability, and trained and tested the model.

But unfortunately, the car always failed to drive at some point around track 1 using my model.  At last i figured out it's because the simulator sends RGB images to the model, but during training i use cv2.imread to load the images, which gives BGR format image data, after fix this problem, the trained model is able to steer the car around track 1 without leaving the road for at least one loop, but at some point the car is very close to the lane line. 

Then i collect some anticlockwise driving data and fine-tune the last model i trained, the resulted model is able to steer the car very well around track 1 without leaving the road.

####2. Final Model Architecture

The final model architecture is a modified version of the [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), please refer to table 1 and the code for details.


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first download the data provided by udaciy as a base, and then i recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_lane_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from getting too close to the lane lines.
These images show what a recovery looks like from close to the left lane to the center of the road :

![alt text][recover_1]
![alt text][recover_2]
![alt text][recover_3]

Then I record some data while driving the car in an oppsite direction. 

I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would make the model has better generalization. For example, here is an image that has then been flipped:

![alt text][original]
![alt text][flipped]

After the collection process, I had about 27K number of data points(including the udacity provided data).


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by that the loss stopped reduce after about 6 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
