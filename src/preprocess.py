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
    print("here?")
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

    return word_frames_dict, all_frames

def read_audios(video_file):
    audio_file = os.path.splitext(video_file)[0] + '.wav'
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file)
    audio_clip.close()
    video_clip.close()

def read_video(video_filepath):
    
    vid = imageio.get_reader(video_filepath, 'ffmpeg')
    return vid


def read_files(src):
    video_name_to_data = {}
    maxi = 20

    for i in range(maxi):
        directory = os.listdir(src)[i]
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

    return video_name_to_data

def load_data(data_folder):
    data_dir = './data/pretrain/'
    window_size = 20

    video_name_to_data = read_files(data_dir)
    video_name_to_data_cleaned = {}
    for name in video_name_to_data.keys():
        if len(video_name_to_data[name][0]) < window_size:
            video_name_to_data_cleaned[name] = video_name_to_data[name]


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

    print("before preprocess")
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

    min_freq = 2
    unk_captions(train_captions, min_freq)
    unk_captions(test_captions, min_freq)

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
            zeros = np.zeros((pad_length, *v[0].shape))
            if pad_length > 0:
                videos[i] = np.concatenate((v, zeros), axis=0)
    
    pad_videos(train_videos, 200)
    pad_videos(test_videos, 200)
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