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
