import os
import cv2
import pickle
import random
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

########################################
# CONFIGURE THESE AS PER YOUR REQUIREMENT
classes = 5
img_w = 96
img_h = 96

BATCH_SIZE = 32
INIT_LR = 1e-3
EPOCHS = 5
########################################

model = models.Sequential()
# First conv -> relu -> pool
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_h, img_w, 3)))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Second conv -> relu -> conv -> relu -> pool
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# third conv -> relu -> conv -> relu -> pool
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))

# Dense -> Softmax
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classes, activation='softmax'))

print(model.summary())

# Read the image paths and their corresponding labels
imagePaths = []
dataset_dir = './cnn-keras/dataset'
for root, dir, files in os.walk(dataset_dir):
    for img in files:
        if img.endswith('.jpg') or img.endswith('.png'):
            imagePaths.append(os.path.join(root, img))

random.seed(55)
random.shuffle(imagePaths)

data = []
labels = []
for img_file in imagePaths:
    img = cv2.imread(img_file)
    img = cv2.resize(img, (img_w, img_h))
    # img = img_to_array(img)
    data.append(img)
    labels.append(img_file.split('/')[-2])

print(len(data), len(labels))

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(X_train, X_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=55)
# print('len x_train, len x_test', len(X_train), len(X_test))
# print('shape X_train', X_train.shape, X_train.ndim)
# print('shape X_test', X_test.shape, X_test.ndim)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode='nearest')


print('compiling model...')
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print('training network...')
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=BATCH_SIZE), 
          validation_data=(X_test, y_test),
          steps_per_epoch=len(X_train) // BATCH_SIZE,
          epochs=EPOCHS, verbose=1)

model.save('./first_model.h5')
f = open('./label_bin','wb')
f.write(pickle.dumps(lb))
f.close()

plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('./plot.png')
