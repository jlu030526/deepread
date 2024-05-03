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
    for file in os.listdir(src):
        if '.txt' in file:
            video_name = os.path.splitext(src+file)[0]
            video_filepath = os.path.splitext(src+file)[0] + ".mp4"
            text_filepath = os.path.splitext(src+file)[0] + ".txt"

            first_line = None
            with open(text_filepath, 'r') as f:
                first_line = f.readline().split()[1:]

            data = pd.read_csv(
                src + file, sep=" ", 
                usecols=["WORD", "START", "END", "ASDSCORE"],
                skiprows=3)
            
            video_name_to_data[video_name] = (first_line, data)

            if len(data) == 0:
                print("WHAT THE F")

    return video_name_to_data

def read_file(file_path):
    data = pd.read_csv(
            file_path, sep=" ", 
            usecols=["WORD", "START", "END", "ASDSCORE"],
            skiprows=3)
    
    return data

def load_data(data_folder):
    data_dir = './data/lrs2/sample/'
    # vid_file_path = data_dir + '00001.mp4'
    # data_file_path = data_dir + '00001.txt'

    # data = read_file(data_file_path)
    # preprocess(data_dir, vid_file_path, data)

    video_name_to_data = read_files(data_dir)
    # print(video_data_pairs)
    data_representations = []

    def get_data_from_names(video_names):
        captions = []
        videos = []
        video_word_mappings = []
        for video_name in video_names:
            video = read_video(video_name + '.mp4')
            word_frame_dict, all_frames = preprocess(data_dir, video, video_name_to_data[video_name][1])
            data_representations.append((video_name_to_data[video_name][0], word_frame_dict))

            captions.append(video_name_to_data[video_name][0])
            videos.append(all_frames)
            video_word_mappings.append(word_frame_dict)
        return videos, captions, video_word_mappings

    shuffled_images = list(video_name_to_data.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    test_image_names = shuffled_images[:len(shuffled_images) // 2]
    train_image_names = shuffled_images[len(shuffled_images) // 2:]

    test_videos, test_captions, test_video_mappings = get_data_from_names(test_image_names)
    train_videos, train_captions, train_video_mappings = get_data_from_names(train_image_names)

    word_count = collections.Counter()
    for caption in train_captions:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_captions, 10)
    unk_captions(test_captions, 10)


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
        train_video_data = train_video_mappings,
        word2idx = word2idx,
        idx2word = {v:k for k,v in word2idx.items()},
    )


def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    # print(f'Data has been dumped into {data_folder}/data.p!')

if __name__ == '__main__':
    data_dir = './data/lrs2/sample/'
    create_pickle(data_dir)