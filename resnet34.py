# models.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import keras_resnet.models

def resnet34_model(input_shape=(224,224,3), num_classes=4, learning_rate=1e-3):
    """
    Xây ResNet-34 pretrained trên ImageNet, bỏ top, rồi add head custom.
    """
    inputs = Input(shape=input_shape)
    # include_top=False để bỏ classifier mặc định
    base = keras_resnet.models.ResNet34(inputs, include_top=False, weights='imagenet')
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model
