import tensorflow as tf

import os
import tensorflow as tf
import numpy as np

class Model(tf.keras.Model):
    def __init__(self):
        

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.cnn_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(),
            tf.kears.layers.MaxPool2D()
        ])

        self.flatten = tf.keras.layers.Flatten()

        self.rnn_layer = tf.keras.Sequential([
            tf.keras.layers.LSTM(),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.LSTM(),
            tf.keras.layers.Dropout(.2)
        ])

        self.ff_layer = tf.keras.layers.Sequential([
            tf.keras.layers.Dense(units=0, activation='relu'),
            tf.keras.layers.Dense(units=0, activation='relu'),
        ])


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        l1_out = self.conv_step(inputs, self.kernel1, False, self.conv_args_layer1, **self.conv_fns_layer1)
        l2_out = self.conv_step(l1_out, self.kernel2, False, self.conv_args_layer2, **self.conv_fns_layer2)
        l3_out = self.conv_step(l2_out, self.kernel3, False, self.conv_args_layer3, **self.conv_fns_layer3)
        l4_out = self.conv_step(l3_out, self.kernel4, is_testing, self.conv_args_layer4, **self.conv_fns_layer4)
        # stu_output = output
        # true_output = tf.nn.conv2d(x, kernel, **conv_args)
        # np.testing.assert_allclose(stu_output, true_output, rtol=1e-5, atol=1e-8)
        output = tf.reshape(l4_out, (l4_out.shape[0], -1)) # make it 2D for linear layers

        l1_out = self.linear_step(output, self.weights1, self.bias1)
        l2_out = self.linear_step(l1_out, self.weights2, self.bias2)
        l3_out = self.linear_step(l2_out, self.weights3, self.bias3, use_dropout=False, use_relu=False)
        prbs = tf.nn.softmax(l3_out)
        
        return prbs

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
        self.loss_list.append(loss)
        return loss

def train(model, train_inputs, train_labels):
    #train model
    pass

def test(model, test_inputs, test_labels):
    #test model
    pass


if __name__ == '__main__':
    main()

