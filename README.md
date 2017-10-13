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
[image6]: ./examples/test.jpg "Test"

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

####3. 

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

![alt text][image6] 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

