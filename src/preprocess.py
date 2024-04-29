import tensorflow as tf
import numpy as np
import os
# import imutils
# import dlib # run "pip install dlib"
import cv2
from PIL import Image

#We could reference:
#https://github.com/deepconvolution/LipNet/blob/master/codes/8_Preprocessing_model.ipynb

def preprocess_image(image_path, width=100, height=100):
    """
    
    """
    
    # with Image.open(image_path) as img:
    #     img_array = np.array(img.resize((100, 100)))


    src = cv2.imread(image_path) 

    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # src.resize((width, height))
    gray_image.resize((width, height))
    # print(gray_image.shape)

    # return src
    return gray_image


def preprocess():
    """
    loop per person ID e.g. F01
        loop per phrase ID e.g. 01
            array representing video frames
            label for array
            loop per instance of phrase (img) e.g. 01
                video = []
                for each image in video:
                    crop img to lips
                    turn img to np.array
                    add to video
        loop per word ID e.g. 01
            array representing video frames
            label for array
            loop per instance of word (img) e.g. 01
                crop img to lips
                turn img to np.
                
    """
    phrase_label_dict = {
        "01": "stop navigation",
        "02": "excuse me",
        "03": "I am sorry",
        "04": "thank you",
        "05": "good bye",
        "06": "i love this game",
        "07": "nice to meet you",
        "08": "you are welcome",
        "09": "how are you?",
        "10": "have a good time",
        }
    word_label_dict = {
        "01": "begin",
        "02": "choose",
        "03": "connection",
        "04": "navigation",
        "05": "next",
        "06": "previous",
        "07": "start",
        "08": "stop",
        "09": "hello",
        "10": "web",
    }

    max_seq_length = 22
    videos = []
    labels = []

    dataset_dir = "data/kaggle/dataset/dataset"
    cropped_dir = "data/kaggle/cropped/cropped"
    
    category = "words"
    person_IDs = os.listdir(cropped_dir)
    path = dataset_dir + "/"
    for person_id in person_IDs:
        print(f"person_ID:{person_id}")
        if (os.path.isdir(dataset_dir + "/" + person_id)):
            phrase_IDs = os.listdir(dataset_dir + "/" + person_id+"/" + category)
            for phrase_id in phrase_IDs:
                if (os.path.isdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id)):
                    instance_IDs = os.listdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id)
                    for instance_id in instance_IDs:
                        frames = []
                        
                        # label = phrase_label_dict[str(instance_id)]

                        if (os.path.isdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id + "/" + instance_id)):
                            label = int(str(instance_id)) - 1

                            for img_path in os.listdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id + "/" + instance_id):
                                dir_path = dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id + "/" + instance_id
                                if "color" in img_path:
                                    img = preprocess_image(dir_path + "/" + img_path)

                                    # print(img)
                                    frames.append(img)


                        # print(frames.shape)
                        # if len(frames) == 10:
                        pad_array = [np.zeros((100, 100))]                            
                        frames.extend(pad_array * (max_seq_length - len(frames)))
                        frame_array = np.asarray(frames)
                        # print(frame_array.shape)
                        label_array = np.asarray(label)
                        # print(label)
                        videos.append(frame_array)
                        labels.append(label_array)
                        

                        # frame_array = np.asarray(frames)
                        # print(frame_array.shape)
                        # label_array = np.asarray(label)
                        # print(label_array.shape)

                        # videos.append(frame_array)
                        # labels.append(label_array)
    # print(len(videos[1]))
    video_array = np.asarray(videos, dtype=object)
    print(video_array.shape)
    bound = int(.8 * len(video_array))
    video_train = video_array[:bound]
    video_test = video_array[bound:]
    video_array = [video_train, video_test]

    labels_array = np.asarray(labels, dtype=object)
    print(labels_array.shape)
    labels_train = labels_array[:bound]
    labels_test = labels_array[bound:]
    labels_array = [labels_train, labels_test]
    # print(video_array.shape)
    
    save_file_names=[
        'output/crop_videos_train.npy',
        'output/crop_videos_test.npy',
        'output/crop_labels_train.npy',
        'output/crop_labels_test.npy']
    np.save(save_file_names[0], video_train, allow_pickle=True)
    np.save(save_file_names[1], video_test, allow_pickle=True)
    np.save(save_file_names[2], labels_train, allow_pickle=True)
    np.save(save_file_names[3], labels_test, allow_pickle=True)
            

def open_data():
    
    X_train = np.load("output/videos_train.npy", allow_pickle=True)
    X_test = np.load("output/videos_test.npy", allow_pickle=True)
    Y_train = np.load("output/labels_train.npy", allow_pickle=True)
    Y_test = np.load("output/labels_test.npy", allow_pickle=True)

    return X_train, Y_train, X_test, Y_test

def open_crop_data():
    X_train = np.load("output/crop_videos_train.npy", allow_pickle=True)
    X_test = np.load("output/crop_videos_test.npy", allow_pickle=True)
    Y_train = np.load("output/crop_labels_train.npy", allow_pickle=True)
    Y_test = np.load("output/crop_labels_test.npy", allow_pickle=True)

    return X_train, Y_train, X_test, Y_test

def load_data(batch_size, buffer_size=1024):
    X_train, Y_train, X_test, Y_test = open_crop_data()

    X_train /= 255.0
    X_test /= 255.0
    # print(X_train[0])

    def normalize_it(X):
        v_min = X.min(axis=(2, 3), keepdims=True)
        v_max = X.max(axis=(2, 3), keepdims=True)
        X = (X - v_min)/(v_max - v_min)
        X = np.nan_to_num(X)
        return X

    # X_train = normalize_it(X_train)
    # X_test = normalize_it(X_test)

    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
    Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)
    X_train = np.expand_dims(X_train, axis=4)
    X_test = np.expand_dims(X_test, axis=4)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return train_dataset, test_dataset

def main():
    preprocess()

if __name__ == '__main__':
    main()

# open_data()
# preprocess()

