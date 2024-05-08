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
            tf.keras.layers.Conv3D(5, (2, 2, 2), strides = 1, input_shape=(200, 160, 160, 1), activation='relu', padding='valid'), 
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            tf.keras.layers.Conv3D(10, (3, 3, 3), strides = 1, activation='relu', padding='valid'),
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.hidden_size)
        ])

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)

        self.flatten = tf.keras.layers.Flatten()

        self.rnn_layer = tf.keras.layers.GRU(self.hidden_size, return_sequences=True)

        self.ff_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=self.hidden_size, activation='relu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(units=self.vocab_size)
        ])


    def call(self, inputs, captions):
        video_embedding = self.cnn_layer(tf.expand_dims(inputs, -1))

        caption_embedding = self.embedding(captions)
        outputs = self.rnn_layer(caption_embedding, initial_state=video_embedding)
        logits = self.ff_layer(outputs)
        return logits

    
    def accuracy_function(self, prbs, labels, mask):
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
    
    def test(self, test_captions, test_videos, test_video_mappings, padding_index, batch_size=30):
        avg_loss = 0
        avg_acc = 0
        avg_prp = 0     
        num_batches = int(len(test_captions) / batch_size)


        indices = tf.range(test_captions.shape[0])

        shuffled_indices = tf.random.shuffle(indices)

        test_captions = tf.gather(test_captions, shuffled_indices)
        test_videos = tf.gather(test_videos, shuffled_indices)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):
            start = end - batch_size
            batch_videos = test_videos[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

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

def save_model(model, dest):
    model.save(dest)

def main():
    data = load_from_pickle('./data')
    model_path = './output/lipReaderModel.keras'

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = Model(len(data["idx2word"].keys()), hidden_size=32, window_size=20)


    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005), 
    )

    epochs = 3

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
        print(f"epoch {e}, test acc{test_acc}")

    model.save('./output/lipReaderModel.keras')


if __name__ == '__main__':
    main()

