# import tensorflow as tf

# import os
# import numpy as np
# from preprocess import load_data

# class Model(tf.keras.Model):
#     def __init__(self):
        
#         super().__init__()
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
#         self.cnn_layer = tf.keras.Sequential([
#             tf.keras.layers.Conv3D(32, (3, 3, 3), strides = 1, input_shape=(22, 100, 100, 1), activation='relu', padding='valid'), 
#             tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
#             tf.keras.layers.Conv3D(64, (3, 3, 3), strides = 1, activation='relu', padding='valid'),
#             tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
#             tf.keras.layers.Conv3D(64, (2, 2, 2), activation='relu', strides=2),
#             tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2)
#         ])

#         self.embedding = tf.keras.layers.Embedding(10, 5) # 5 is hyperparam

#         self.flatten = tf.keras.layers.Flatten()

#         self.rnn_layer = tf.keras.Sequential([
#             tf.keras.layers.LSTM(32, return_sequences=True),
#             tf.keras.layers.Dropout(.2),
#         ])

#         self.ff_layer = tf.keras.Sequential([
#             # tf.keras.layers.Dense(units=128, activation='relu'),
#             # tf.keras.layers.Dropout(.2),
#             tf.keras.layers.Dense(units=64, activation='relu'),
#             tf.keras.layers.Dropout(.2),
#             tf.keras.layers.Dense(units=10),
#         ])


#     def call(self, inputs, captions):
#         outputs = inputs
#         # print(outputs.shape)
#         video_embedding = self.cnn_layer(outputs)
#         # print
#         video_embedding = tf.reshape(outputs, (outputs.shape[0],outputs.shape[-1], outputs.shape[2]*outputs.shape[3]))

#         caption_embedding = self.embedding(captions)
#         outputs = self.rnn_layer(caption_embedding, hidden_state=video_embedding)
#         outputs = self.flatten(outputs)
#         logits = self.ff_layer(outputs)

#         # image_embedding = self.cnn_layer(inputs)
#         # word_embedding = self.embedding(caption)
#         return logits

    
#     def accuracy_function(self, prbs, labels):
#         # mask = tf.fill(labels.shape, True)
#         # correct_classes = tf.argmax(prbs, axis=-1) == labels
#         # boolean_vals = tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask)
#         # print((list(boolean_vals)))
#         # accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask), keepdims=True)

#         mask = tf.cast(labels >= 0, tf.bool)  # Assuming labels >= 0 for valid labels
#         correct_classes = tf.equal(tf.argmax(prbs, axis=-1), labels)
#         # print(correct_classes)
#         correct = tf.where(correct_classes).shape[0]
#         # accuracy = tf.reduce_mean(tf.cast(correct_classes, tf.float32), axis=0)
#         # print(labels.shape)
#         # print(correct)
#         accuracy = correct/labels.shape[0]
#         return accuracy

# # def train(model, train_inputs, train_labels):
# def train(model, trainloader):
#     # indices = tf.range(start=0, limit=tf.shape(train_inputs)[0], dtype=tf.int32)
#     # shuffled_indices = tf.random.shuffle(indices)

#     # train_inputs = tf.gather(train_inputs, shuffled_indices)
#     # train_labels = tf.gather(train_labels, shuffled_indices)
#     accuracy_sum = 0
#     iterations = 0
#     # for b, b1 in enumerate(range(model.batch_size, train_inputs.shape[0] + 1, model.batch_size)):
#     for i, batch in enumerate(trainloader):
#         # print(f"iteration:{i}")
#         # print("here")
#         X = batch[0]
#         Y = batch[1]
#         # b0 = b1 - model.batch_size
#         with tf.GradientTape() as tape:
#             y_pred = model((X))
#             # print(y_pred, Y)
#             Y = tf.cast(Y, tf.int64)
#             loss = model.loss(y_pred, Y)
#             # print('loss: ', loss)
#             acc = model.accuracy_function(y_pred, Y)
#             accuracy_sum += acc
#         gradients = tape.gradient(loss, model.trainable_variables)
#         # print(gradients)
#         model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


#     average_accuracy = accuracy_sum / i * 1.0
#     return average_accuracy

# def embedding_train(model, trainloader):
#     for batch in trainloader:
#         X = batch[0]
#         Y = batch[1]
#         iterations += 1
#         # b0 = b1 - model.batch_size
#         with tf.GradientTape() as tape:
#             y_pred = model(X)
#             loss = model.loss(y_pred, Y)
#             acc = model.accuracy(y_pred, Y)
#             accuracy_sum += acc
#         gradients = tape.gradient(loss, model.trainable_variables)
#         model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     average_accuracy = accuracy_sum / iterations * 1.0
#     return average_accuracy



# def test(model, testloader):
#     accuracy_sum = 0
#     iterations = 0
#     # for b, b1 in enumerate(range(model.batch_size, test_inputs.shape[0] + 1, model.batch_size)):
#     for batch in testloader:
#         iterations += 1
#         # b0 = b1 - model.batch_size
#         X = batch[0]
#         Y = batch[1]
#         Y = tf.cast(Y, tf.int64)
#         y_pred = model(X)

#         acc = model.accuracy_function(y_pred, Y)
#         accuracy_sum += acc

#     average_accuracy = accuracy_sum / iterations * 1.0
#     return average_accuracy

# def main():
#     train_dataset, test_dataset = load_data(batch_size=5)
#     # print(train_dataset, test_dataset)
#     # train_inputs = train_dataset[0]
#     # train_labels = train_dataset[1]
#     # test_inputs = test_dataset[0]
#     # test_labels = test_dataset[1]

#     model = Model()
#     model.compile(loss=loss_fn)

#     epochs=10
#     for e in range(epochs):
#         train_acc = train(model, train_dataset)
#         print(f"epoch:{e}, train_acc:{train_acc}")

#     # print('accuracy: ', train_acc)
#     acc = test(model, test_dataset)
#     print(f"test_acc:{acc}")

# def loss_fn(prbs, labels):
#     # scce = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=True)
#     labels = tf.keras.utils.to_categorical(labels, num_classes=10)
#     scce = tf.keras.losses.CategoricalCrossentropy()(labels, prbs)
#     return scce

# if __name__ == '__main__':
#     main()


import tensorflow as tf

import os
import numpy as np
from preprocess import load_from_pickle

class Model(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, window_size):

        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.cnn_layer = tf.keras.Sequential([
            tf.keras.layers.Conv3D(32, (3, 3, 3), strides = 1, input_shape=(125, 160, 160, 1), activation='relu', padding='valid'), 
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            tf.keras.layers.Conv3D(64, (3, 3, 3), strides = 1, activation='relu', padding='valid'),
            # tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            # tf.keras.layers.Conv3D(64, (2, 2, 2), activation='relu', strides=2),
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            tf.keras.layers.Dense(units=self.hidden_size, activation='relu'),
            tf.keras.layers.Dense(units=self.hidden_size)
            
        ])

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size) # 5 is hyperparam

        self.flatten = tf.keras.layers.Flatten()

        self.rnn_layer = tf.keras.Sequential([
            tf.keras.layers.LSTM(self.hidden_size, return_sequences=True),
            tf.keras.layers.Dropout(.2),
        ])

        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.hidden_size, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(units=self.vocab_size),
        ])


    def call(self, inputs, captions):
        # print(outputs.shape)

        video_embedding = self.cnn_layer(tf.expand_dims(inputs, -1))
        # print
        # video_embedding = tf.reshape(video_embedding, (video_embedding.shape[0],video_embedding.shape[-1], video_embedding.shape[2]*video_embedding.shape[3]))

        caption_embedding = self.embedding(captions)
        outputs = self.rnn_layer(caption_embedding, hidden_state=video_embedding)
        outputs = self.flatten(outputs)
        logits = self.ff_layer(outputs)

        # image_embedding = self.cnn_layer(inputs)
        # word_embedding = self.embedding(caption)
        return logits

    
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
    
    def loss_function(self, probs, decoder_labels):
        loss_metric = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(probs, decoder_labels)
        return loss_metric

    def train(self, train_captions, train_videos, train_word_mappings, padding_index, batch_size=30):

        avg_loss = 0
        avg_acc = 0
        avg_prp = 0     
        num_batches = int(len(train_captions) / batch_size)

        # get rainge of train_captions
        indices = tf.range(train_captions.shape[0])

        # get shufffled indices
        shuffled_indices = tf.random.shuffle(indices)

        # gather traincaptions, indices
        train_captions = tf.gather(train_captions, shuffled_indices)
        # gather with image_features, shuffledcaptions
        train_videos = tf.gather(train_videos, shuffled_indices)
        # train_videos = tf.gather(train_videos, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        print(batch_size, len(train_captions))
        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):
            start = end - batch_size
            batch_videos = train_videos[start:end, :]
            decoder_input = train_captions[start:end, :-1]
            decoder_labels = train_captions[start:end, 1:]

            ## Perform a training forward pass. Make sure to factor out irrelevant labels.
            with tf.GradientTape() as tape:
                probs = self(batch_videos, decoder_input)
                num_predictions = decoder_labels != padding_index
                loss = self.loss_function(probs, decoder_labels)
                print(loss)
                accuracy = self.accuracy_function(probs, decoder_labels)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            ## Compute and report on aggregated statistics
            total_loss += loss

            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()
        return avg_loss, avg_acc, avg_prp
    
    def test(self, test_captions, train_videos, train_video_mappings, batch_size=30):
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()        
        return avg_prp, avg_acc


def main():
    data = load_from_pickle('./data')

    print(len(data["idx2word"].keys()))
    model = Model(len(data["idx2word"].keys()), 5, 20)

    def perplexity(labels, preds):
        entropy = tf.keras.metrics.sparse_categorical_crossentropy(labels, preds, from_logits=False, axis=-1) 
        entropy = tf.reduce_mean(entropy)
        perplexity = tf.exp((entropy))
        return perplexity 
    acc_metric  = perplexity

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005), 
        metrics=[acc_metric],
    )

    epochs = 5

    for e in range(epochs):
        train_acc = model.train(
            tf.convert_to_tensor(data["train_captions"]), 
            tf.convert_to_tensor(data["train_videos"]), 
            data["word2idx"]['<pad>'],
            data["train_video_mappings"], batch_size=15)
        print(f"epoch:{e}, train_acc:{train_acc}")

    # model = Model()
    # model.compile(loss=loss_fn)

    # epochs=10
    # for e in range(epochs):
    #     train_acc = model.train(model, data["train_captions"], data["train_videos"], data["train_video_data"])
    #     print(f"epoch:{e}, train_acc:{train_acc}")

    # # print('accuracy: ', train_acc)
    # acc = model.test(model, data["test_captions"], data["test_videos"], data["test_video_mappings"])
    # print(f"test_acc:{acc}")

def loss_fn(prbs, labels):
    # scce = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=True)
    labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    scce = tf.keras.losses.CategoricalCrossentropy()(labels, prbs)
    return scce

if __name__ == '__main__':
    main()

