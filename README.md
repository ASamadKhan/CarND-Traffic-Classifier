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
* As a first step, I decided to convert the images to grayscale using cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) function

* In the 2nd step of preprocessing I normalized the images using  cv2.normalize(image,dest ,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
after applying the above 2 prepeocessing steps the images are changed in the below figure 

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
* validation set accuracy of **92.5%** 
* test set accuracy of **90** 

### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

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

####2. 

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

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This may be due to the clear signs in these images.

#### 3. 
### We are going to discuss the softmax probailities of only 3 examples out of 8  
For this image 
![alt text][image9] <br />
our model predicted pedestrains with high probability (0.9) while General Caution which is little closer sign is predicted with probability .019.  


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.80994523e-01      			| pedestrians  									| 
| 1.90053992e-02    				| General caution										|
| 1.16753204e-10				| Right-of-way at the next intersection											|
| .76334190e-11	      			| Road narrows on the right					 				|
| 9.66769416e-13			    | Traffic signals      							|



For this image 
![alt text][image13] <br />
our model bicycles crossing  with high probability (0.997) while Beware of ice/snow	 which is little closer sign is predicted with probability .00237.  




| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.97115135e-01      			| bicycles crossing  									| 
| 2.37373519e-03   				| Beware of ice/snow									|
| 2.72262609e-04				| Children crossing										|
| 1.29659209e-04      			| Road narrows on the right				 				|
| 7.18888477e-05		    | Slippery road     							| 


For this image 
![alt text][image15] <br />
our model Priority road  with high probability (0.997) 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99835849e-01    			| Priority road 									| 
| 1.63932928e-04  				| Roundabout mandatory									|
| 9.56486303e-08				| End of no passing by vehicles over	3.5 metric tons								|
| 7.05458518e-08      			|Right-of-way at the next intersection			 				|
| 3.18835021e-08		    | End of no passing     							| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

