import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import sys

from keras_preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint


checkpoint = ModelCheckpoint("cnn_gems.h5", 
                    monitor="acc",
                    verbose=1,
                    mode="max",
                    period=1,
                    save_weights_only=False,
                    save_best_only=True)

img_datagen = ImageDataGenerator(rescale=1/255,  
                   rotation_range=45,
                   horizontal_flip=True,
                   vertical_flip=True, 
                   width_shift_range=0.2,
                   height_shift_range=0.2,
                   )

img_generator = img_datagen.flow_from_directory(
    './data',
    target_size=(96, 96),
    batch_size=64,
    class_mode='categorical'
)

def train_first():
    model = MobileNetV2(input_shape=(96, 96, 3),
                                 weights='imagenet',
                                 include_top=False)
    
    x_ = model.output
    x_ = GlobalAveragePooling2D()(x_)
    x_ = Dense(1024, activation='relu')(x_)
    preds = Dense(286, activation='softmax')(x_)  
    
    model = Model(inputs = model.input, outputs=preds)
    model.compile(optimizer=SGD( momentum=0.9, nesterov=True),
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])

    return model

def train_again():
    return load_model("cnn_gems.h5")

def main(argc, argv):
    if argc != 2:
        exit()

    if argv[1] == '-ta':
        model=train_again()
    elif argv[1] == '-tf':
        model=train_first()


    model.fit_generator(img_generator, callbacks=[checkpoint],
                     steps_per_epoch=2966/64, epochs=20)

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)












