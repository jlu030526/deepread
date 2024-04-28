import tensorflow as tf

import os
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
            tf.keras.layers.Dense(units=0, activation='softmax'),
        ])


    def call(self, inputs):
        outputs = inputs
        outputs = self.cnn_layer(outputs)
        outputs = self.flatten(outputs)
        outputs = self.rnn_layer(outputs)
        outputs = self.ff_layer(outputs)
        return outputs

    def loss(self, logits, labels):
        # TODO: figure out a loss function
        pass

def train(model, train_inputs, train_labels):
    indices = tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    train_inputs = tf.gather(train_inputs, shuffled_indices)
    train_labels = tf.gather(train_labels, shuffled_indices)
    accuracy_sum = 0
    iterations = 0
    for b, b1 in enumerate(range(model.batch_size, train_inputs.shape[0] + 1, model.batch_size)):
        iterations += 1
        b0 = b1 - model.batch_size
        with tf.GradientTape() as tape:
            y_pred = model(augment_data(train_inputs[b0:b1]), is_testing=True)
            loss = model.loss(y_pred, train_labels[b0:b1])
            acc = model.accuracy(y_pred, train_labels[b0:b1])
            accuracy_sum += acc
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    average_accuracy = accuracy_sum / iterations * 1.0
    return average_accuracy

def augment_data(inputs):
    inputs = tf.image.random_flip_left_right(inputs)
    return inputs

def test(model, test_inputs, test_labels):
    accuracy_sum = 0
    iterations = 0
    for b, b1 in enumerate(range(model.batch_size, test_inputs.shape[0] + 1, model.batch_size)):
        iterations += 1
        b0 = b1 - model.batch_size
        y_pred = model(test_inputs[b0:b1])

        acc = model.accuracy(y_pred, test_labels[b0:b1])
        accuracy_sum += acc

    average_accuracy = accuracy_sum / iterations * 1.0
    return average_accuracy


# if __name__ == '__main__':
#     main()

