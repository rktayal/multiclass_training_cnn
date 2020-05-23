# multiclass_training_cnn
The repo has source code for CNN network and initiating the training on any dataset for multi class classification problem. 

The repo can act as a guide for someone who wants to implement a CNN classifier from scratch
The repo has 3 source code files:
 - network.py
 - train.py
 - classify.py

`network.py` defines the CNN network architecture. 
it expects input image of size (96, 96, 3). However that can be modified
to any image size in the source code.
NOTE: Since the network is not very deep, if you are increase the input image size
to say (200x200x3), chances are, you may not get good accuracy. 
It outputs a vector having probabilites of all the classes present. (class with
maximum probability becomes the predicted label for the image)

The architecture has following layers:

```
             Conv layer with Relu Activation and batch normalization (32 filters of size 3x3)
             Pooling layer with Dropout (pooling window of 2x2, stride 2)
                                  |
                                 \|/
             Conv layer with Relu Activation and batch normalization (64 filters of size 3x3)
             Conv layer with Relu Activation and batch normalization (64 filters of size 3x3)
             Pooling layer with Dropout (pooling window of 2x2, stride 2)
                                  |
                                 \|/
             Conv layer with Relu Activation and batch normalization (128 filters of size 3x3)
             Conv layer with Relu Activation and batch normalization (128 filters of size 3x3)
             Pooling layer with Dropout followed by flattened (pooling window of 2x2, stride 2)
                                  |
                                 \|/
             Dense layer with RELU activation (1024 neurons)
             Dense layer with number of nuerons = number of classes and softmax activation

Detailed Network architecture looks like this:


Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 94, 94, 32)        896       
_________________________________________________________________
batch_normalization_1 (Batch (None, 94, 94, 32)        128       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 47, 47, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 47, 47, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 45, 45, 64)        18496     
_________________________________________________________________
batch_normalization_2 (Batch (None, 45, 45, 64)        256       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 43, 43, 64)        36928     
_________________________________________________________________
batch_normalization_3 (Batch (None, 43, 43, 64)        256       
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 21, 21, 64)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 21, 21, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 19, 19, 128)       73856     
_________________________________________________________________
batch_normalization_4 (Batch (None, 19, 19, 128)       512       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 17, 17, 128)       147584    
_________________________________________________________________
batch_normalization_5 (Batch (None, 17, 17, 128)       512       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 8, 8, 128)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 128)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 8192)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              8389632   
_________________________________________________________________
batch_normalization_6 (Batch (None, 1024)              4096      
_________________________________________________________________
dropout_4 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 5125      
=================================================================
Total params: 8,678,277
Trainable params: 8,675,397
Non-trainable params: 2,880
```

### Training
To perfrom training, all we have to do is execute the train script.
Before executing, you need to change following lines in `train.py` file
line 15-26 in `train.py`
```
########################################
# CONFIGURE THESE AS PER YOUR REQUIREMENT
dataset_dir = './dataset'   # change as per your dataset directory
classes = 5                 # change as per no. of classes in your dataset
img_w = 96                  # image width you want to feed to network.
img_h = 96                  # image height you want to feed to network. 
img_d = 3

BATCH_SIZE = 32             # Batch size, you can keep it as 32 for starters. No need to change
INIT_LR = 1e-3              # learning rate, 0.0001 (you can keep it as this only)
EPOCHS = 5                  # number of epochs to train, you can keep it as it is for starters.
########################################
```
Once you have made the changes, you can initiate the training using the command
```
python train.py
```
The script first read all the paths of the images and put it in a list.
Then it loads all the images present in the above created list of image files and
extract the label for each image from its filename.
To convert string label into one hot encoding form, `LabelBinarizer` class is used present
in `sklearn.preprocessing` package.
These two arrays i.e. data (containing all images) and labels (containing label for each image)
are then split into a train, test set using `train_test_split` method of `sklearn.model_processing`
Using keras API, a model is compiled with Adam optimizer and
training is initiated using `fit_generator` api
Once the training is complete, model is dumped in the same directory with the name `first_model.h5`
which can be loaded afterwads and used for inferencing (`classify.py` script does that)

Why we have used categorical cross-entropy as our loss function for this problem?
Input: two vectors usually one is ground truth vector and other is predicted vector from network.
Outputs: Real number i.e. cross-entropy which we will use as our loss value

### Inference
To perform inference, you can execute the `classify.py` script
before executing, you need to change the following lines in `classify.py`
Line-8 in `classify.py`
```
filename = './image_to_be_classified.jpg'
```
Once you have made the change, you can perform classification using the command
```
python classify.py
```
The script first loads the image, resizes it to what model is expecting, perform 
inference, as in doing a forward pass using the `predict` api of keras. 
it will give us a vector of probabilites for each class. We fetch the index of class
having maximum probability score using `numpy.argmax` and that becomes our predicted label/class.
Next, we display the image along with the predicted label for visual purposes

### For curious readers, intution behind choosing categorical crossentropy loss during training
Say we have classification problem with 3 classes: cat, dog, bird
<br />

Sample 1:
for a sample image of cat, our ground truth vector will look like [1, 0, 0]
Now, let’s assume vector our model predicted for that image is [0.6, 0.3, 0.1] 
<br />

Sample 2:
for a sample image of cat, our ground truth vector will look like [1, 0, 0]
Now, let’s assume vector our model predicted for that image is [0.2, 0.7, 0.1] 
<br />

Sample 3
for another sample image of cat, our ground truth vector will look like [1, 0, 0]
Now, let’s assume the vector our model predicted for that image is [1, 0, 0] 

#### Requirement
We want to penalise our model more in second sample since the predicted probability 
for that class (0.2) is way lesser as compared to the prediction in first sample (0.6)
hence loss should be more in case of sample 2 as compared to loss value in case of sample 1
<br />

In case of sample 3, since our model predicted exactly correct, we don’t want to penalise 
our model at all.  Hence loss should be zero in this case.
<br />

Therefore, a cross-entropy of 0.0 when training a model indicates that the predicted class 
probabilities are identical to the probabilities in the training dataset, e.g. zero loss.
<br />

Categorical cross entropy serves our above requirements. <br />
Let's consider categorical cross-entropy function: <br />
L(y, y´) = -∑ y * log(y´) (summation for all elements in vector, in our case 3) <br />
where y is ground truth vector and y´ is predicted vector <br />

```
Calclulating L(y,y´) for Sample 1:
y = [1, 0, 0]
y´= [0.6, 0.3, 0.1]
L(y, y´) = - (1 * log(0.6) + 0 * log(0.3) + 0 * log(0.1))
L(y, y´) = - (-0.593 + 0 + 0)
L(y, y´) = 0.510

Calclulating L(y,y´) for Sample 2:
y = [1, 0, 0]
y´= [0.2, 0.7, 0.1]
L(y, y´) = - (1 * log(0.2) + 0 * log(0.7) + 0 * log(0.1))
L(y, y´) = - (-1.609 + 0 + 0)
L(y, y´) = 1.609

Calclulating L(y,y´) for Sample 3:
y = [1, 0, 0]
y´= [1, 0, 0]
L(y, y´) = - (1 * log(1) + 0 * log(0) + 0 * log(0))
L(y, y´) = - (0 + 0 + 0)
L(y, y´) = 0

Since loss value in Sample 2(1.609) was more as compared to loss in sample 1 (0.510),
Therefore above function serves our purpose of penalizing bad predictions more.
If the probability associated with the true class is 1.0, we need its loss to be zero. 
Conversely, if that probability is low, say, 0.01, we need its loss to be HUGE!

When log function is applied on values greater than 1, it acts like attenuator. 
This property is hugely utilized in plotting data that has too huge range (say 20 to 30000). 
Well in those situations, we simply plot the data on log-scale.

When log function is applied to small values (smaller than 1), we see another useful behavior. 
In that region, it acts like a negative magnifier.

since the log of values between 0.0 and 1.0 is negative, 
we take the negative log to obtain a positive value for the loss
```
### Intution behind choosing softmax activation function for last layer
Since our problem statement is a multi class classification problem meaning
where only one result can be correct. In other words, an example can belong to one class only.
<br />

The softmax function highlights the largest values and suppresses values which are 
significantly below the maximum value, though this is not true for small values. 
It normalizes the outputs so that they sum to 1 so that they can be directly treated as probabilities over the output.
```
Going back to our 3 class classification problem
if we have final layer vector as [2.4, 1.0, 4.5] then
applying softmax will return us vector of same dimension
and the values will reflect the probabilites of each class. 
The sum of all values will be 1. <br />
Softmax function will supress values which are below the maximum value
as mentioned above. <br />
[2.4, 1.0, 4.5] ----Softmax function----> [0.12, 0.02, 0.86]
hence output label becomes bird since index 3 has max probability. <br />

```
How it is calculated?
[2.4, 1.0, 4.5] --> [y1, y2, y3]
y1 = e^2.4 / (e^2.4 + e^1.0 + e^4.5) = 0.12
y2 = e^1.0 / (e^2.4 + e^1.0 + e^4.5) = 0.02
y3 = e^4.5 / (e^2.4 + e^1.0 + e^4.5) = 0.86

In general, softmax func = e^x / (summation of e^x) <br />
where x is each element in vector
```
```

