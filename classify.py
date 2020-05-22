import cv2
import pickle
import imutils
import numpy as np
from keras import models
from keras.preprocessing.image import img_to_array

filename = './cnn-keras/examples/bulbasaur_plush.png'
img = cv2.imread(filename)
output = img.copy()
img = cv2.resize(img, (96, 96))
img = img.astype('float')/255.0
img = img_to_array(img)
img = np.expand_dims(img, axis=0)

model = models.load_model('./first_model.h5')
lb = pickle.loads(open('./label_bin', 'rb').read())

prob = model.predict(img)[0]
idx = np.argmax(prob)
label = lb.classes_[idx]

f = filename[filename.rfind('/')+1:]
correct = "correct" if f.rfind(label) != -1 else "incorrect"
label = "{}: {:.2f}% ({})".format(label, prob[idx] * 100, correct)
ouput = imutils.resize(output, width=400)

cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)
