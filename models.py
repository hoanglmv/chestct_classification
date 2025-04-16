from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                                     Dropout, Dense, GlobalAveragePooling2D)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def cnn_model(input_shape=(224,224,3), num_classes=4, weight_decay=1e-4):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(16, 3, activation="relu", kernel_regularizer=l2(weight_decay)),
        MaxPooling2D(),
        Conv2D(32, 3, activation="relu", kernel_regularizer=l2(weight_decay)),
        MaxPooling2D(),
        Conv2D(64, 3, activation="relu", kernel_regularizer=l2(weight_decay)),
        MaxPooling2D(),
        Conv2D(128,3, activation="relu", kernel_regularizer=l2(weight_decay)),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu", kernel_regularizer=l2(weight_decay)),
        Dropout(0.15),
        Dense(64, activation="relu", kernel_regularizer=l2(weight_decay)),
        Dropout(0.15),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(1e-3),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model

def vgg_model(input_shape=(224,224,3), num_classes=4):
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(1e-3),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model
