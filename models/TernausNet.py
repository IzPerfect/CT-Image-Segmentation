import keras
from keras import models
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Activation, Lambda, Conv2D, \
    MaxPooling2D, UpSampling2D,Input, Concatenate, Conv2DTranspose
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from data_utils import *

class TernausNet(object):
    def __init__(self, img_shape, num_of_class, actf = 'relu',
        learning_rate = 0.001):

        '''
        Arguments :

        img_shape - shape of input image (64, 64, 1)
        actf - activation function for network training
        learning_rate - learning rate for training
        '''

        self.learning_rate = learning_rate
        self.actf = actf
        self.img_shape = img_shape
        self.num_of_class = num_of_class

        self.model = self.build_model()

    # encoding block
    def enc_conv_block(self, x, layers, feature_maps, filter_size = (3, 3),
                           conv_strides = 1, pooling_filter_size = (2, 2), pooling_strides = (2, 2)):
        for i in range(layers):
            x = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                               padding = 'same', kernel_initializer = 'he_normal')(x)

        pool = MaxPooling2D(pooling_filter_size, strides = pooling_strides)(x)

        # pool : next input, x : copy and concatenate
        return pool, x

    # decoding block
    def dec_conv_block(self, inputs, merge_inputs, layers, feature_maps, trans_feature_maps, filter_size = (3, 3), conv_strides = 1,
                           up_conv_strides = (2, 2)):

        merge = Concatenate(axis = 3)([Conv2DTranspose(trans_feature_maps, filter_size,
                                                       activation = self.actf, strides = up_conv_strides, kernel_initializer = 'he_normal',
                                                       padding = 'same')(inputs), merge_inputs])
        x = merge
        for i in range(layers):
            x = Conv2D(feature_maps , filter_size , activation = self.actf, strides = conv_strides,
                               padding = 'same', kernel_initializer = 'he_normal')(x)
        return x

    # encoder
    def encoding_path(self, inputs):

        enc_conv1, concat1 = self.enc_conv_block(inputs, 1, 64)
        enc_conv2, concat2 = self.enc_conv_block(enc_conv1, 1, 128)
        enc_conv3, concat3 = self.enc_conv_block(enc_conv2, 2, 256)
        enc_conv4, concat4 = self.enc_conv_block(enc_conv3, 2, 512)
        enc_conv5, concat5 = self.enc_conv_block(enc_conv4, 2, 512)

        return concat1, concat2, concat3, concat4, concat5, enc_conv5

    # decoder
    # In decoding_path, the filter outputs is half of the previos
    def decoding_path(self, dec_inputs, concat1, concat2, concat3, concat4, concat5):

        dec_conv1 = self.dec_conv_block(dec_inputs, concat5, 1, 512, 256)
        dec_conv2 = self.dec_conv_block(dec_conv1, concat4, 1, 512, 256)
        dec_conv3 = self.dec_conv_block(dec_conv2, concat3, 1, 256, 128)
        dec_conv4 = self.dec_conv_block(dec_conv3, concat2, 1, 128, 64)
        dec_conv5 = self.dec_conv_block(dec_conv4, concat1, 1, 64, 32)

        return dec_conv5
    # build network
    def build_model(self):
        inputs = Input(self.img_shape)

        # Contracting path
        concat1, concat2, concat3, concat4, concat5, enc_path = self.encoding_path(inputs)

        # center
        center = Conv2D(512, (3,3), activation = self.actf, padding = 'same', kernel_initializer = 'he_normal')(enc_path)

        # Expanding path
        dec_path = self.decoding_path(center, concat1, concat2, concat3, concat4, concat5)
        segmented = Conv2D(self.num_of_class, (1,1), activation ='sigmoid', padding = 'same', kernel_initializer = 'glorot_normal')(dec_path)

        model = Model(inputs = inputs, outputs = segmented)
        model.compile(optimizer = Adam(lr = self.learning_rate),
                          loss = 'binary_crossentropy', metrics = [dice_coef])

        return model

    # train model
    def train(self, X_train, Y_train, epoch = 10, batch_size = 32, val_split = 0.2, shuffle = True):

        self.history = self.model.fit(X_train, Y_train, validation_split = val_split,
                                          epochs = epoch, batch_size = batch_size, shuffle =  shuffle)
        return self.history

    # predict test data
    def predict(self, X_test):
        pred_classes = self.model.predict(X_test)

        return pred_classes

    # show u-net architecture
    def show_model(self):
        return print(self.model.summary())
