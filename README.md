#**Traffic Sign Recognition** 
---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization1.jpg "Visualization1"
[image2]: ./examples/visualization2.jpg "Visualization2"
[image3]: ./examples/visualization3.jpg "Visualization3"
[image4]: ./examples/classes.jpg "classes"
[image5]: ./examples/grey.jpg "greyclasses"
[image6]: ./examples/8.jpg "Test"
[image7]: ./examples/8O.jpg.jpg "Test"
[image8]: ./examples/3.jpg "Test"
[image9]: ./examples/3O.jpg "Test"
[image10]: ./examples/1.jpg "Test"
[image11]: ./examples/1O.jpg "Test"
[image12]: ./examples/2.jpg "Test"
[image13]: ./examples/2O.jpg "Test"
[image14]: ./examples/4.jpg "Test"
[image15]: ./examples/4O.jpg "Test"
[image16]: ./examples/7.jpg "Test"
[image17]: ./examples/7O.jpg "Test"
[image18]: ./examples/5.jpg "Test"
[image19]: ./examples/5O.jpg "Test"
[image20]: ./examples/6.jpg "Test"
[image21]: ./examples/6O.jpg "Test"
## Rubric Points

---

### Data Set Summary & Exploration


* The size of training set is = **34799**
* The size of the validation set is = **4410** 
* The size of test set is = **12630**
* The shape of a traffic sign image is = **(32, 32, 3)**
* The number of unique classes/labels in the data set is = **43**

### Include an exploratory visualization of the dataset.

* These bar chart shows class distributio in traiing validation and test data sets. We can easily see the distributions are almost same in all of them.

![Test Data][image1]
![alt text][image2]
![alt text][image3]

* Calasses with captioning is generated in the below diagram.

![alt text][image4]

### Design and Test a Model Architecture
### 1.
* As a first step, I decided to convert the images to grayscale using cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) function. This function convert colored image into gray image.

* In the 2nd step of preprocessing I normalized the images using  followig code. <br />
<p align = "center"> X_train = (train_gray-128)/128  </p> 
<p align = "center"> X_valid = (valid_gray-128)/128 </p>  
<p align = "center"> X_test = (test_gray-128)/128 </p>   

since the gray scale values are from 0-255, so we subtract 128 from each pixel and then dividing it by 128. This process will convert all pixel values from -1 to 1.
![alt text][image5]



#### 2. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5x6     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 ksize, 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16	    | 1x1 stride, Valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 ksize, 2x2 stride,  outputs 5x5x16 				|
| Fullyconnected	layer_0	| The feature space is falttend with first layer has 400 neurons|
| dropout	| dropout with keep_prop=0.6 is applied to reduce overfittting |
| Fullyconnected	layer_1	| 400x120 this layer has 120 neurons|
| RELU					|												|
| Fullyconnected	layer_2	| 120x84 this layer has 84 neurons|
| RELU					|												|
| Fullyconnected	layer_3	| 84x43 this layer has 43 neurons |
| Softmax				| softmax converts these scores into 43 class probabilities|

The output from the Softmax are 43 class probabilities.   

#### 3. 

To train the model, I used an admamoptimizer with learnig rate of 0.001. The number of epoches were 20 and the batch size is 128

#### 4. 
I first tried the lenet architecture with colores images but the test accuracy and validation accuracy did not improve. I also let the lenet learn for more than 20 epoches but the overfitting was observed. Tried some different learning rates too but the overall result did not imorove much.

My final model results were:
* training set accuracy of **99%**
* validation set accuracy of **95%** 
* test set accuracy of **94%** 

### Test a Model on New Images

#### 1. 

I found 8 images of traffic signs on the the web.

![alt text][image6] <br />
![alt text][image7] <br />
![alt text][image8] <br />
![alt text][image9] <br />
![alt text][image10] <br />
![alt text][image11] <br />
![alt text][image12] <br />
![alt text][image13] <br />
![alt text][image14] <br />
![alt text][image15] <br />
![alt text][image16] <br />
![alt text][image17] <br />
![alt text][image18] <br />
![alt text][image19] <br />
![alt text][image20] <br />
![alt text][image21] <br />

#### 2. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield      		| Yield  									| 
| Pedestrians     			| Pedestrians										|
| Vehicles over 3.5 metric tons prohibited					| Vehicles over 3.5 metric tons prohibited										|
| Bicycles crossing	      		| Bicycles crossing					 				|
| Priority road		| Priority road      							|
| Turn right ahead		| Turn right ahead      							|
| Road work	| Road work    							|
| Stop	| Stop      							|

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. <br />
**The images are very straight forward with no  rotation, scaling and cropping**

#### 3. 
### We are going to discuss the softmax probailities of only 3 examples out of 8  
For this image 
![alt text][image9] <br />
our model predicted pedestrains with high probability (0.98) while General Caution which is little closer sign is predicted with probability .013.  


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.85018790e-01      			| pedestrians  									| 
| 1.34794246e-02    				| General caution										|
| 1.50138198e-03				| Right-of-way at the next intersection											|
| 1.67010498e-07	      			| Road narrows on the right					 				|
| 1.41821104e-07		    | Traffic signals      							|



For this image 
![alt text][image13] <br />
our model bicycles crossing  with high probability (0.99) while Beware of ice/snow	 which is little closer sign is predicted with probability .00068.  


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99164224e-01      			| bicycles crossing  									| 
| 6.81320380e-04   				| Beware of ice/snow									|
| 1.11647794e-04				| Children crossing										|
| 2.90547414e-05     			| Road narrows on the right				 				|
| 7.84349686e-06		    | Slippery road     							| 


For this image 
![alt text][image15] <br />
our model Priority road  with high probability (0.994) 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.94987965e-01  			| Priority road 									| 
| 3.59199126e-03  				| Roundabout mandatory									|
| 1.36826362e-03				| End of no passing by vehicles over	3.5 metric tons								|
| 1.64698558e-05      			|Right-of-way at the next intersection			 				|
| 1.41892415e-05		    | End of no passing     							| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

