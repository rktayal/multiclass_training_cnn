from keras import models
from keras import layers


def get_network(width, height, depth, classes):
    input_shape = (height, width, depth)

    def build_model():
        model = models.Sequential()
        # First conv -> relu -> pool
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
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
        return model
    return build_model

