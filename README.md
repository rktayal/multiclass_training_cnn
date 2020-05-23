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

### Inference
