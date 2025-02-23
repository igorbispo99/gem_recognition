import sys

from keras_preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint


checkpoint = ModelCheckpoint("cnn_gems.h5", 
                    monitor="val_acc",
                    verbose=1,
                    mode="max",
                    period=1,
                    save_weights_only=False,
                    save_best_only=True)

img_datagen = ImageDataGenerator(rescale=1/255,  
                   rotation_range=45,
                   horizontal_flip=True,
                   vertical_flip=True, 
                   )

train_gen = img_datagen.flow_from_directory(
    './data/train',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)

val_gen = img_datagen.flow_from_directory(
    './data/test',
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical'
)


def train_first():
    base_model = MobileNetV2(input_shape=(224, 224, 3),
                                 weights='imagenet',
                                 include_top=False)
    
    x_ = base_model.output
    x_ = GlobalAveragePooling2D()(x_)
    preds = Dense(87, activation='softmax')(x_)  
    
    model = Model(inputs = base_model.input, outputs=preds)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

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


    model.fit_generator(train_gen, callbacks=[checkpoint],
                     steps_per_epoch=train_gen.samples // 16,
                     epochs=10, validation_data = val_gen,
                     validation_steps = val_gen.samples // 16 )

if __name__ == "__main__":
    main(len(sys.argv), sys.argv)












