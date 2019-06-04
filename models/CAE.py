import keras
from keras import models
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from data_utils import *

class CAE(object):
    def __init__(self, img_shape, actf = 'relu',
        learning_rate = 0.001,  drop_rate = 0.5, do_batch_norm = False, do_drop = False):
        '''
        Arguments :

        img_shape - shape of input image (64, 64, 1)
        actf - activation function for network training
        learning_rate - learning rate for training
        drop_rate - dropout rate
        do_batch_norm - whether to run for batchnormalization
        do_drop - whether to run for dropout
        '''

        self.learning_rate = learning_rate
        self.actf = actf
        self.img_shape = img_shape
        self.drop_rate = drop_rate
        self.do_batch_norm = do_batch_norm
        self.do_drop = do_drop

        self.model = self.build_model()

    # just enter input function
    def cae_data_input(self, x):
        return x

    # input shape
    def cae_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


    # encoder part
    def encoding_path(self, model):
        # encoder layer1
        model.add(Conv2D(16, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization()) if self.do_batch_norm else None
        model.add(Activation(self.actf))
        model.add(MaxPooling2D((2,2), strides = (2,2)))

        # encoder layer2
        model.add(Conv2D(32, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization()) if self.do_batch_norm else None
        model.add(Activation(self.actf))
        model.add(MaxPooling2D((2,2), strides = (2,2)))

        # encoder layer 3
        model.add(Conv2D(64, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization()) if self.do_batch_norm else None
        model.add(Activation(self.actf))
        model.add(MaxPooling2D((2,2), strides = (2,2)))

        return model

    # decoder part
    def decoding_path(self, model):
        # decoder layer1
        model.add(Conv2D(64, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization()) if self.do_batch_norm else None
        model.add(Activation(self.actf))
        model.add(UpSampling2D((2,2)))

        # decoder layer2
        model.add(Conv2D(32, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization()) if self.do_batch_norm else None
        model.add(Activation(self.actf))
        model.add(UpSampling2D((2,2)))

        # decoder layer3
        model.add(Conv2D(16, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(BatchNormalization()) if self.do_batch_norm else None
        model.add(Activation(self.actf))
        model.add(UpSampling2D((2,2)))

        model.add(Conv2D(1, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(Activation('sigmoid'))

        return model

    # build network
    def build_model(self):
        model = models.Sequential()

        #  input layer
        model.add(Lambda(self.cae_data_input, input_shape = self.img_shape))

        # encoding
        model = self.encoding_path(model)

        # middle layer
        model.add(Conv2D(128, (3,3), strides = 1,  padding = 'same', kernel_initializer = 'he_normal'))
        model.add(Dropout(self.drop_rate)) if self.do_drop else None

        # decoding
        model = self.decoding_path(model)

        model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = self.learning_rate)
                          , metrics = [dice_coef])
        return model

    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2):

        self.history = self.model.fit(X_train, Y_train, validation_split = val_split,
                                          epochs = epoch, batch_size = batch_size)
        return self.history

    # predict test data
    def predict(self, X_test):
        pred_classes = self.model.predict(X_test)

        return pred_classes

    # show model architecture
    def show_model(self):
        return print(self.model.summary())
