import tensorflow as tf

import os
import numpy as np
from preprocess import load_data

class Model(tf.keras.Model):
    def __init__(self):
        
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.cnn_layer = tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (3, 3, 3), strides = 1, input_shape=(10, 100, 100, 1), activation='relu', padding='valid'), 
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            tf.keras.layers.Conv3D(64, (3, 3, 3), strides = 1, activation='relu', padding='valid'),
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            # tf.keras.layers.Conv3D(64, (2, 2, 2), activation='relu', strides=2),
            # tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2)
        ])

        self.embedding = tf.keras.layers.Embedding(10, 5) # 5 is hyperparam

        self.flatten = tf.keras.layers.Flatten()

        self.rnn_layer = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(.2),
        ])

        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(units=10),
        ])


    def call(self, inputs):
        outputs = inputs
        outputs = self.cnn_layer(outputs)
        outputs = tf.reshape(outputs, (outputs.shape[0],outputs.shape[-1], outputs.shape[2]*outputs.shape[3]))
        outputs = self.rnn_layer(outputs)
        outputs = self.flatten(outputs)
        outputs = self.ff_layer(outputs)

        # image_embedding = self.cnn_layer(inputs)
        # word_embedding = self.embedding(caption)
        return outputs

    
    def accuracy_function(self, prbs, labels):
        # mask = tf.fill(labels.shape, True)
        # correct_classes = tf.argmax(prbs, axis=-1) == labels
        # boolean_vals = tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask)
        # print((list(boolean_vals)))
        # accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask), keepdims=True)

        mask = tf.cast(labels >= 0, tf.bool)  # Assuming labels >= 0 for valid labels
        correct_classes = tf.equal(tf.argmax(prbs, axis=-1), labels)
        # print(correct_classes)
        correct = tf.where(correct_classes).shape[0]
        # accuracy = tf.reduce_mean(tf.cast(correct_classes, tf.float32), axis=0)
        # print(labels.shape)
        # print(correct)
        accuracy = correct/labels.shape[0]
        return accuracy

# def train(model, train_inputs, train_labels):
def train(model, trainloader):
    # indices = tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32)
    # shuffled_indices = tf.random.shuffle(indices)

    # train_inputs = tf.gather(train_inputs, shuffled_indices)
    # train_labels = tf.gather(train_labels, shuffled_indices)
    accuracy_sum = 0
    iterations = 0
    # for b, b1 in enumerate(range(model.batch_size, train_inputs.shape[0] + 1, model.batch_size)):
    for i, batch in enumerate(trainloader):
        # print(f"iteration:{i}")
        # print("here")
        X = batch[0]
        Y = batch[1]
        # b0 = b1 - model.batch_size
        with tf.GradientTape() as tape:
            y_pred = model((X))
            # print(y_pred, Y)
            Y = tf.cast(Y, tf.int64)
            loss = model.loss(y_pred, Y)
            # print('loss: ', loss)
            acc = model.accuracy_function(y_pred, Y)
            accuracy_sum += acc
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    average_accuracy = accuracy_sum / i * 1.0
    return average_accuracy

def embedding_train(model, trainloader):
    for batch in trainloader:
        X = batch[0]
        Y = batch[1]
        iterations += 1
        # b0 = b1 - model.batch_size
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = model.loss(y_pred, Y)
            acc = model.accuracy(y_pred, Y)
            accuracy_sum += acc
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    average_accuracy = accuracy_sum / iterations * 1.0
    return average_accuracy



def test(model, testloader):
    accuracy_sum = 0
    iterations = 0
    # for b, b1 in enumerate(range(model.batch_size, test_inputs.shape[0] + 1, model.batch_size)):
    for batch in testloader:
        iterations += 1
        # b0 = b1 - model.batch_size
        X = batch[0]
        Y = batch[1]
        Y = tf.cast(Y, tf.int64)
        y_pred = model(X)

        acc = model.accuracy_function(y_pred, Y)
        accuracy_sum += acc

    average_accuracy = accuracy_sum / iterations * 1.0
    return average_accuracy

def main():
    train_dataset, test_dataset = load_data(batch_size=25)
    # print(train_dataset, test_dataset)
    # train_inputs = train_dataset[0]
    # train_labels = train_dataset[1]
    # test_inputs = test_dataset[0]
    # test_labels = test_dataset[1]

    model = Model()
    model.compile(loss=loss_fn)

    epochs=10
    for e in range(epochs):
        train_acc = train(model, train_dataset)
        print(f"epoch:{e}, train_acc:{train_acc}")

    # print('accuracy: ', train_acc)
    acc = test(model, test_dataset)
    print(f"test_acc:{acc}")

def loss_fn(prbs, labels):
    # scce = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=True)
    labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    scce = tf.keras.losses.CategoricalCrossentropy()(labels, prbs)
    return scce

if __name__ == '__main__':
    main()

