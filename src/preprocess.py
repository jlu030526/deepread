# import tensorflow as tf
# import numpy as np
# import os
# # import imutils
# # import dlib # run "pip install dlib"
# # import cv2
# from PIL import Image

# #We could reference:
# #https://github.com/deepconvolution/LipNet/blob/master/codes/8_Preprocessing_model.ipynb

# def preprocess_image(image_path, width=100, height=100):
#     """
    
#     """
    
#     # with Image.open(image_path) as img:
#     #     img_array = np.array(img.resize((100, 100)))


#     src = cv2.imread(image_path) 

#     gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#     # src.resize((width, height))
#     gray_image.resize((width, height))
#     # print(gray_image.shape)

#     # return src
#     return gray_image


# def preprocess():
#     """
#     loop per person ID e.g. F01
#         loop per phrase ID e.g. 01
#             array representing video frames
#             label for array
#             loop per instance of phrase (img) e.g. 01
#                 video = []
#                 for each image in video:
#                     crop img to lips
#                     turn img to np.array
#                     add to video
#         loop per word ID e.g. 01
#             array representing video frames
#             label for array
#             loop per instance of word (img) e.g. 01
#                 crop img to lips
#                 turn img to np.
                
#     """
#     phrase_label_dict = {
#         "01": "stop navigation",
#         "02": "excuse me",
#         "03": "I am sorry",
#         "04": "thank you",
#         "05": "good bye",
#         "06": "i love this game",
#         "07": "nice to meet you",
#         "08": "you are welcome",
#         "09": "how are you?",
#         "10": "have a good time",
#         }
#     word_label_dict = {
#         "01": "begin",
#         "02": "choose",
#         "03": "connection",
#         "04": "navigation",
#         "05": "next",
#         "06": "previous",
#         "07": "start",
#         "08": "stop",
#         "09": "hello",
#         "10": "web",
#     }

#     max_seq_length = 22
#     videos = []
#     labels = []

#     dataset_dir = "data/kaggle/dataset/dataset"
#     cropped_dir = "data/kaggle/cropped/cropped"
    
#     category = "words"
#     person_IDs = os.listdir(cropped_dir)
#     path = dataset_dir + "/"
#     for person_id in person_IDs:
#         print(f"person_ID:{person_id}")
#         if (os.path.isdir(dataset_dir + "/" + person_id)):
#             phrase_IDs = os.listdir(dataset_dir + "/" + person_id+"/" + category)
#             for phrase_id in phrase_IDs:
#                 if (os.path.isdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id)):
#                     instance_IDs = os.listdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id)
#                     for instance_id in instance_IDs:
#                         frames = []
                        
#                         # label = phrase_label_dict[str(instance_id)]

#                         if (os.path.isdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id + "/" + instance_id)):
#                             label = int(str(instance_id)) - 1

#                             for img_path in os.listdir(dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id + "/" + instance_id):
#                                 dir_path = dataset_dir + "/" + person_id + "/"+category+"/" + phrase_id + "/" + instance_id
#                                 if "color" in img_path:
#                                     img = preprocess_image(dir_path + "/" + img_path)

#                                     # print(img)
#                                     frames.append(img)


#                         # print(frames.shape)
#                         # if len(frames) == 10:
#                         pad_array = [np.zeros((100, 100))]                            
#                         frames.extend(pad_array * (max_seq_length - len(frames)))
#                         frame_array = np.asarray(frames)
#                         # print(frame_array.shape)
#                         label_array = np.asarray(label)
#                         # print(label)
#                         videos.append(frame_array)
#                         labels.append(label_array)
                        

#                         # frame_array = np.asarray(frames)
#                         # print(frame_array.shape)
#                         # label_array = np.asarray(label)
#                         # print(label_array.shape)

#                         # videos.append(frame_array)
#                         # labels.append(label_array)
#     # print(len(videos[1]))
#     video_array = np.asarray(videos, dtype=object)
#     print(video_array.shape)
#     bound = int(.8 * len(video_array))
#     video_train = video_array[:bound]
#     video_test = video_array[bound:]
#     video_array = [video_train, video_test]

#     labels_array = np.asarray(labels, dtype=object)
#     print(labels_array.shape)
#     labels_train = labels_array[:bound]
#     labels_test = labels_array[bound:]
#     labels_array = [labels_train, labels_test]
#     # print(video_array.shape)
    
#     save_file_names=[
#         'output/crop_videos_train.npy',
#         'output/crop_videos_test.npy',
#         'output/crop_labels_train.npy',
#         'output/crop_labels_test.npy']
#     np.save(save_file_names[0], video_train, allow_pickle=True)
#     np.save(save_file_names[1], video_test, allow_pickle=True)
#     np.save(save_file_names[2], labels_train, allow_pickle=True)
#     np.save(save_file_names[3], labels_test, allow_pickle=True)
            

# def open_data():
    
#     X_train = np.load("output/videos_train.npy", allow_pickle=True)
#     X_test = np.load("output/videos_test.npy", allow_pickle=True)
#     Y_train = np.load("output/labels_train.npy", allow_pickle=True)
#     Y_test = np.load("output/labels_test.npy", allow_pickle=True)

#     return X_train, Y_train, X_test, Y_test

# def open_crop_data():
#     X_train = np.load("output/crop_videos_train.npy", allow_pickle=True)
#     X_test = np.load("output/crop_videos_test.npy", allow_pickle=True)
#     Y_train = np.load("output/crop_labels_train.npy", allow_pickle=True)
#     Y_test = np.load("output/crop_labels_test.npy", allow_pickle=True)

#     return X_train, Y_train, X_test, Y_test

# def load_data(batch_size, buffer_size=1024):
#     X_train, Y_train, X_test, Y_test = open_crop_data()

#     X_train /= 255.0
#     X_test /= 255.0
#     # print(X_train[0])

#     def normalize_it(X):
#         v_min = X.min(axis=(2, 3), keepdims=True)
#         v_max = X.max(axis=(2, 3), keepdims=True)
#         X = (X - v_min)/(v_max - v_min)
#         X = np.nan_to_num(X)
#         return X

#     # X_train = normalize_it(X_train)
#     # X_test = normalize_it(X_test)

#     X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    
#     X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
#     Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
#     Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float32)
#     X_train = np.expand_dims(X_train, axis=4)
#     X_test = np.expand_dims(X_test, axis=4)

#     train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
#     train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

#     test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
#     test_dataset = test_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

#     return train_dataset, test_dataset

# def main():
#     preprocess()

# if __name__ == '__main__':
#     main()

# # open_data()
# # preprocess()


import tensorflow as tf
import numpy as np
import os
import pandas as pd
import imageio
# from pylab import *
from moviepy.editor import VideoFileClip
import math
import pickle
import collections
import random


def preprocess_captions(captions):
    for i, caption in enumerate(captions):
        # Join those words into a string
        caption_new = ['<start>'] + caption + ['<end>']
      
        # Replace the old caption in the captions list with this new cleaned caption
        captions[i] = caption_new


def preprocess(output_dir, video, timestamp_data):
    words = timestamp_data["WORD"]
    start = timestamp_data["START"]
    end = timestamp_data["END"]

    word_frames_dict = {}
    all_frames = []
    for word, start, end in zip(words, start, end):
        
        frames = []
        frame_num = math.ceil(start * video.get_meta_data()['fps'])
        while frame_num < math.floor(end * video.get_meta_data()['fps']):
            frame = video.get_data(frame_num)
            gray_frame = np.dot(frame[:,:,:3], [0.2989, 0.5870, 0.1140])
            frames.append(gray_frame)
            frame_num += 1
        
        word_frames_dict[word] = frames
        all_frames += frames

        # out_file = output_dir+word+".mp4"
        # ffmpeg_extract_subclip(video, start, end, targetname=out_file)

    return word_frames_dict, all_frames

def read_audios(video_file):
    audio_file = os.path.splitext(video_file)[0] + '.wav'
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file)
    audio_clip.close()
    video_clip.close()

    # audio = wave.open(audio_file, 'rb')

def read_video(video_filepath):
    
    vid = imageio.get_reader(video_filepath, 'ffmpeg')

    # for i in range(len(video.get_meta_data()))

    # container = av.open(video_filepath)

    # ims = [frame.to_image() for frame in container.decode(video=0)]
    # ims = [im.convert('L') for im in ims ]
    # ims = np.array([np.array(im) for im in ims])
    # print(ims.shape)
    # print(vid)
    return vid


def read_files(src):
    video_name_to_data = {}
    iteration = 0
    for directory in os.listdir(src):
        for file in os.listdir(src + '' + directory):
            if '.txt' in file:
                video_name = os.path.splitext(src + directory + '/' + file)[0]
                text_filepath = video_name + '.txt'

                first_line = None
                with open(text_filepath, 'r') as f:
                    first_line = f.readline().split()[1:]

                data = pd.read_csv(
                    text_filepath, sep=" ", 
                    usecols=["WORD", "START", "END", "ASDSCORE"],
                    skiprows=3)
                
                video_name_to_data[video_name] = (first_line, data)

                if len(data) == 0:
                    print("WHAT THE F")

        iteration += 1
        if iteration == 4:
            break

    return video_name_to_data

def read_file(file_path):
    data = pd.read_csv(
            file_path, sep=" ", 
            usecols=["WORD", "START", "END", "ASDSCORE"],
            skiprows=3)
    
    return data

def load_data(data_folder):
    data_dir = './data/pretrain/'
    window_size = 20
    # vid_file_path = data_dir + '00001.mp4'
    # data_file_path = data_dir + '00001.txt'

    # data = read_file(data_file_path)
    # preprocess(data_dir, vid_file_path, data)

    video_name_to_data = read_files(data_dir)
    video_name_to_data_cleaned = {}
    for name in video_name_to_data.keys():
        if len(video_name_to_data[name][0]) < window_size:
            video_name_to_data_cleaned[name] = video_name_to_data[name]

    # print(video_data_pairs)
    data_representations = []

    def get_data_from_names(video_names):
        captions = []
        videos = []
        video_word_mappings = []
        for video_name in video_names:
            video = read_video(video_name + '.mp4')
            word_frame_dict, all_frames = preprocess(data_dir, video, video_name_to_data_cleaned[video_name][1])
            data_representations.append((video_name_to_data_cleaned[video_name][0], word_frame_dict))

            captions.append(video_name_to_data_cleaned[video_name][0])
            videos.append(all_frames)
            video_word_mappings.append(word_frame_dict)
        return videos, captions, video_word_mappings

    shuffled_images = list(video_name_to_data_cleaned.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    test_image_names = shuffled_images[:len(shuffled_images) // 2]
    train_image_names = shuffled_images[len(shuffled_images) // 2:]

    test_videos, test_captions, test_video_mappings = get_data_from_names(test_image_names)
    train_videos, train_captions, train_video_mappings = get_data_from_names(train_image_names)

    preprocess_captions(test_captions)
    preprocess_captions(train_captions)

    word_count = collections.Counter()
    for caption in train_captions:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_captions, 5)
    unk_captions(test_captions, 5)

    def pad_captions(captions, window_size):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * ['<pad>'] 
    
    pad_captions(train_captions, window_size)
    pad_captions(test_captions,  window_size)

    def max_frames(videos):
        max_frames = 0
        for video in videos:
            max_frames = max(len(video), max_frames)
        return max_frames + 1

    def pad_videos(videos, max_frames):
        for i, v in enumerate(videos):
            pad_length = max_frames - len(v)
            if pad_length > 0:
                zeros = np.zeros((pad_length, *v[0].shape))
                videos[i] = np.concatenate((v, zeros), axis=0)
    
    pad_videos(train_videos, max_frames(train_videos))
    pad_videos(test_videos, max_frames(test_videos))

    word2idx = {}
    vocab_size = 0
    for caption in train_captions:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in test_captions:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word]

    return dict(
        test_videos = test_videos,
        test_captions = test_captions,
        test_video_mappings = test_video_mappings,
        train_videos = train_videos,
        train_captions = train_captions,
        train_video_mappings = train_video_mappings,
        word2idx = word2idx,
        idx2word = {v:k for k,v in word2idx.items()},
    )


def load_from_pickle(data_file_path):
    with open(f'{data_file_path}/data.p', 'rb') as pickle_file:
        data = pickle.load(pickle_file)

    return data

def create_pickle(data_folder):
    file_to_save = './data'
    with open(f'{file_to_save}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {file_to_save}/data.p!')

if __name__ == '__main__':
    data_dir = './data/lrs2/sample/'
    create_pickle(data_dir)