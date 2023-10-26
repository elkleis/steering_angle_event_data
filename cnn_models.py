"""
Script with all models used for training.
"""

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import add
from keras import regularizers
from keras.applications import ResNet50

regular_constant = 0


def resnet50(img_width, img_height, img_channels, output_dim):
    img_input = Input(shape=(img_height, img_width, img_channels))

    base_model = ResNet50(input_tensor=img_input,
                          weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    # Steering channel
    output = Dense(output_dim)(x)

    model = Model(inputs=img_input, outputs=[output])
    # print(model.summary())

    return model


def resnet50_random_init(img_width, img_height, img_channels, output_dim):
    img_input = Input(shape=(img_height, img_width, img_channels))

    base_model = ResNet50(input_tensor=img_input,
                          weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)

    # Steering channel
    output = Dense(output_dim)(x)

    model = Model(inputs=[img_input], outputs=[output])
    # print(model.summary())

    return model
