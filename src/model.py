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
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
        self.cnn_layer = tf.keras.Sequential([
            tf.keras.layers.Conv3D(5, (2, 2, 2), strides = 1, input_shape=(125, 160, 160, 1), activation='relu', padding='valid'), 
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            tf.keras.layers.Conv3D(10, (3, 3, 3), strides = 1, activation='relu', padding='valid'),
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            # tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            # tf.keras.layers.Conv3D(64, (2, 2, 2), activation='relu', strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.hidden_size)
        ])

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size) # 5 is hyperparam

        self.flatten = tf.keras.layers.Flatten()

        self.rnn_layer = tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)

        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.hidden_size, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(units=self.vocab_size)
        ])


    def call(self, inputs, captions):
        # print(outputs.shape)

        video_embedding = self.cnn_layer(tf.expand_dims(inputs, -1))
        # print
        # video_embedding = tf.reshape(video_embedding, (video_embedding.shape[0],video_embedding.shape[-1], video_embedding.shape[2]*video_embedding.shape[3]))

        caption_embedding = self.embedding(captions)
        outputs = self.rnn_layer(caption_embedding, initial_state=video_embedding)
        # outputs = self.flatten(outputs)
        logits = self.ff_layer(outputs)
        return logits

    
    def accuracy_function(self, prbs, labels, mask):
        # mask = tf.fill(labels.shape, True)
        # correct_classes = tf.argmax(prbs, axis=-1) == labels
        # boolean_vals = tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask)
        # print((list(boolean_vals)))
        # accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask), keepdims=True)

        labels = tf.cast(labels, tf.int64)
        correct_classes = tf.argmax(tf.cast(prbs, tf.float64), axis=-1) == labels
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(correct_classes, tf.float32), mask))
        return accuracy
    
    def loss_function(self, probs, decoder_labels):
        loss = tf.keras.losses.sparse_categorical_crossentropy(decoder_labels, probs, from_logits=True)
        loss = tf.reduce_sum(loss)
        return loss

    def train(self, train_captions, train_videos, train_word_mappings, padding_index, batch_size=30):

        avg_loss = 0
        avg_acc = 0
        avg_prp = 0     
        num_batches = int(len(train_captions) / batch_size)


        indices = tf.range(train_captions.shape[0])

        shuffled_indices = tf.random.shuffle(indices)

        train_captions = tf.gather(train_captions, shuffled_indices)
        train_videos = tf.gather(train_videos, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):
            start = end - batch_size
            batch_videos = train_videos[start:end, :]
            decoder_input = train_captions[start:end, :-1]
            decoder_labels = train_captions[start:end, 1:]

            with tf.GradientTape() as tape:
                probs = self(batch_videos, decoder_input)
                mask = decoder_labels != padding_index
                num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
                loss = self.loss_function(probs, decoder_labels)
                accuracy = self.accuracy_function(probs, decoder_labels, mask)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            total_loss += loss

            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()
        return avg_loss, avg_acc, avg_prp
    
    def test(self, test_captions, train_videos, train_video_mappings, padding_index, batch_size=30):
        avg_loss = 0
        avg_acc = 0
        avg_prp = 0     
        num_batches = int(len(train_captions) / batch_size)


        indices = tf.range(train_captions.shape[0])

        shuffled_indices = tf.random.shuffle(indices)

        train_captions = tf.gather(train_captions, shuffled_indices)
        train_videos = tf.gather(train_videos, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(train_captions)+1, batch_size)):
            start = end - batch_size
            batch_videos = train_videos[start:end, :]
            decoder_input = train_captions[start:end, :-1]
            decoder_labels = train_captions[start:end, 1:]

            probs = self(batch_videos, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)
            
            total_loss += loss

            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()
        return avg_loss, avg_acc, avg_prp


def main():
    data = load_from_pickle('./data')


    model = Model(len(data["idx2word"].keys()), 5, 20)

    # def perplexity(labels, preds):
    #     entropy = tf.keras.metrics.sparse_categorical_crossentropy(labels, preds, from_logits=False, axis=-1) 
    #     entropy = tf.reduce_mean(entropy)
    #     perplexity = tf.exp((entropy))
    #     return perplexity 
    # acc_metric  = perplexity

    ## TODO: Compile your model using your choice of optimizer, loss, and metrics
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005), 
        # metrics=[acc_metric],
    )

    epochs = 5

    for e in range(epochs):
        train_acc = model.train(
            tf.convert_to_tensor(data["train_captions"]), 
            tf.convert_to_tensor(data["train_videos"]), 
            data["train_video_mappings"], data["word2idx"]['<pad>'], batch_size=15)
        print(f"epoch:{e}, train_acc:{train_acc}")
    
    test_acc = model.test(
        tf.convert_to_tensor(data["test_captions"]), 
        tf.convert_to_tensor(data["test_videos"]), 
        data["test_video_mappings"], data["word2idx"]['<pad>'], batch_size=15
    )
    print(f'test acc{test_acc}')

    # model = Model()
    # model.compile(loss=loss_fn)

    # epochs=10
    # for e in range(epochs):
    #     train_acc = model.train(model, data["train_captions"], data["train_videos"], data["train_video_data"])
    #     print(f"epoch:{e}, train_acc:{train_acc}")

    # # print('accuracy: ', train_acc)
    # acc = model.test(model, data["test_captions"], data["test_videos"], data["test_video_mappings"])
    # print(f"test_acc:{acc}")

if __name__ == '__main__':
    main()

