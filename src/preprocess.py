import tensorflow as tf
import numpy as np
import os
# import imutils
# import dlib # run "pip install dlib"
import cv2
from PIL import Image

#We could reference:
#https://github.com/deepconvolution/LipNet/blob/master/codes/8_Preprocessing_model.ipynb

def preprocess_image(image_path):
    """
    
    """
    
    with Image.open(image_path) as img:
        img_array = np.array(img.resize((300, 300)))

    src = cv2.imread(image_path) 
    gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img_array = np.array(gray_image.resize((300, 300)))
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
    videos = []
    labels = []

    dataset_dir = "data/kaggle/dataset/dataset"
    
    person_IDs = os.listdir(dataset_dir)
    path = dataset_dir + "/"
    for person_id in person_IDs[:6]:
        print(f"person_ID:{person_id}")
        if (os.path.isdir(dataset_dir + "/" + person_id)):
            phrase_IDs = os.listdir(dataset_dir + "/" + person_id+"/phrases")
            for phrase_id in phrase_IDs:
                if (os.path.isdir(dataset_dir + "/" + person_id + "/phrases/" + phrase_id)):
                    instance_IDs = os.listdir(dataset_dir + "/" + person_id + "/phrases/" + phrase_id)
                    for instance_id in instance_IDs:
                        frames = []
                        
                        # label = phrase_label_dict[str(instance_id)]

                        if (os.path.isdir(dataset_dir + "/" + person_id + "/phrases/" + phrase_id + "/" + instance_id)):
                            label = phrase_label_dict[str(instance_id)]

                            for img_path in os.listdir(dataset_dir + "/" + person_id + "/phrases/" + phrase_id + "/" + instance_id):
                                dir_path = dataset_dir + "/" + person_id + "/phrases/" + phrase_id + "/" + instance_id
                                if "color" in img_path:
                                    img = preprocess_image(dir_path + "/" + img_path)
                                    # print(img.shape)
                                    frames.append(img)


                        # print(frames.shape)
                        frame_array = np.asarray(frames)
                        label_array = np.asarray(label)

                        videos.append(frame_array)
                        labels.append(label_array)
    # print(len(videos[1]))
    video_array = np.asarray(videos, dtype=object)
    bound = int(.8 * len(video_array))
    video_train = video_array[:bound]
    video_test = video_array[bound:]
    video_array = [video_train, video_test]

    labels_array = np.asarray(labels, dtype=object)
    labels_train = video_array[:bound]
    labels_test = video_array[bound:]
    labels_array = [labels_train, labels_test]
    
    np.save('output/videos.npy', video_array, allow_pickle=True)
    np.save('output/labels.npy', labels_array, allow_pickle=True)
            

def open_data():
    labels = np.load("output/labels.npy", allow_pickle=True)
    bound = int(.8 * labels.shape[0])
    Y_train = labels[:bound]
    Y_test = labels[bound:]
    # print(labels[:3])
    # print(labels.shape)

    videos = np.load("output/videos.npy", allow_pickle=True)
    X_train = videos[:bound]
    X_test = videos[bound:]
    print(videos[:3])
    print(videos[0].shape)

    return X_train, Y_train, X_test, Y_test

def load_data(batch_size, buffer_size=1024):
    X_train, Y_train, X_test, Y_test = open_data()
    X_train /= 255.0
    X_test /= 255.0

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset = test_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return train_dataset, test_dataset



# open_data()
preprocess()

