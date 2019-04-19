# coding=utf-8
'''
@author leeguandong
paper:Total Recall Understanding traffic signs using deep hierarchical convolutional neural networkTotal Recall Understanding traffic signs using deep hierarchical convolutional neural network
'''

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.layers import Input, Multiply, GlobalAveragePooling2D, Add, Dense, Activation
from keras.models import Model, load_model
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.layers.convolutional import *
from keras.models import Model
from keras import callbacks
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
image_size = 224
epoch = 30
train_size = 3000
test_size = 1000
num_classes = 43

##
train_path = ''
test_path = ''
classes = []

train_datagen = ImageDataGenerator(rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rescale=1.0 / 255,
                                   horizontal_flip=False,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_batches = train_datagen.flow_from_directory(train_path, target_size=(image_size, image_size), color_mode='rgb',
                                                  classes=classes, batch_size=batch_size, class_mode='categorical')
test_batches = test_datagen.flow_from_directory(test_path, target_size=(image_size, image_size), color_mode='rgb',
                                                classes=classes, batch_size=batch_size, class_mode='categorical')
X = train_batches


def dilated_block(X, num_channel, base):
    # trunk path
    X_shortcut = X

    X = BatchNormalization(axis=-1, name=base + '/Branch1/bn_1')(X)
    X = Activation('relu')(X)
    X = Conv2D(num_channel, (3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Conv2D(num_channel, (3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    # Branch path1
    X_shortcut1 = BatchNormalization(axis=-1, name=base + '/Branch2/bn_1')(X_shortcut)
    X_shortcut1 = Activation('relu')(X_shortcut1)
    X_shortcut1 = Conv2D(num_channel, (3, 3), strides=(1, 1), dilation_rate=2, padding='same',
                         kernel_initializer=glorot_uniform(seed=0))(X_shortcut1)
    X = Add()([X, X_shortcut1])

    # trunk path
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)
    X = Conv2D(num_channel, (3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)

    # Branch path2
    X_shortcut2 = BatchNormalization(axis=-1)(X_shortcut)
    X_shortcut2 = Activation('relu')(X_shortcut2)
    X_shortcut2 = Conv2D(num_channel, (3, 3), strides=(1, 1), dilation_rate=3, padding='same',
                         kernel_initializer=glorot_uniform(seed=0))(X_shortcut2)
    X = Add()([X, X_shortcut2])

    return X


def CNN_model():
    input_shape = (image_size, image_size, 3)
    X_input = Input(input_shape)

    X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=-1)(X)
    X = Activation('relu')(X)

    X = dilated_block(X, 64, 'Block1')
    X = dilated_block(X, 64, 'Block2')

    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)

    X = dilated_block(X, 64, 'Block3')
    X = dilated_block(X, 128, 'Block4')

    X = MaxPooling2D((3, 3), strides=(1, 1))(X)

    X = Flatten()(X)
    X = Dense(512)(X)
    X = Dense(num_classes)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='Dilated_skipped_model')
    model.summary()
    return model


def callback_for_training(tf_log_dir_name='./tf-log/', patience_lr=2):
    cb = [None] * 3

    # Tensorboard log callback
    tb = callbacks.TensorBoard(log_dir=tf_log_dir_name, histogram_freq=0)
    cb[0] = tb

    # Early stopping callback
    # early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='suto',
    #                                      save_best_only=True)
    # cb.append(early_stop)

    # Model checkpointer
    mdc = callbacks.ModelCheckpoint('bestmodel={epoch:02d}-{val_loss:.2f}.hdf5', verbose=0, monitor='val_acc',
                                    save_best_only=True)
    cb[1] = mdc

    # Reduce learning Rate
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1e-1, patience=patience_lr, verbose=1,
                                                 mode='min', min_delta=1e-4, min_lr=1e-12)
    cb[2] = reduce_lr_loss
    return cb


cb = callback_for_training()
model = CNN_model()
plot_model(model, 'DilatedSkipnetwork.png')
model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_batches, shuffle=True, steps_per_epoch=train_size // batch_size,
                              validation_data=test_batches, validation_steps=test_size // batch_size, epochs=epoch,
                              verbose=1, callbacks=cb)

model.save('GTSRB_Train01.h5')
model = load_model('GTSRB_Train01.h5')


def plot_loss_acc(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'bo')
    plt.plot(epochs, val_loss, 'g')
    plt.title('Training and validation loss')
    plt.legend(['train', 'val'], loc='upper right')
    filename = 'GTSRB_Approach08_loss.png'
    plt.savefig(filename)
    plt.show()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'g')
    plt.title('Training and validation accuracy')
    plt.legend(['train', 'val'], loc='lower right')
    filename = 'GTSRB_Approach08_accuracy.png'
    plt.savefig(filename)
    plt.show()


plot_loss_acc(history)
