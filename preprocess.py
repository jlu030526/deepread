import tensorflow as tf
import numpy as np
from functools import reduce

#We could reference:
#https://github.com/deepconvolution/LipNet/blob/master/codes/8_Preprocessing_model.ipynb

#From HW4 
def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg index->word mapping)
    """
    # Hint: You might not use all of the initialized variables depending on how you implement preprocessing. 
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    ## TODO: Implement pre-processing for the data files. See notebook for help on this.

    # Read and tokenize training data
    with open(train_file, 'r') as training_file:
        train_data = training_file.read().split()
        unique_training_content = list(set(train_data))

    # Read and tokenize testing data
    with open(test_file, 'r') as testing_file:
        test_data = testing_file.read().split()

    # Create vocabulary mapping
    vocabulary = {word: index for index, word in enumerate(unique_training_content)}

    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)
    
    # Uncomment the sanity check below if you end up using vocab_size
    # Sanity check, make sure that all values are withi vocab size
    # assert all(0 <= value < vocab_size for value in vocabulary.values())

    # Vectorize, and return output tuple.
    train_data = [float(vocabulary[word]) for word in train_data]
    test_data = [float(vocabulary[word]) for word in test_data]

    # print("train_data", train_data)
    return train_data, test_data, vocabulary
