import tensorflow as tf
import numpy as np
import os
import pandas as pd
import imageio
# from pylab import *
from moviepy.editor import VideoFileClip
import math
import pickle

def preprocess(output_dir, video, timestamp_data):
    words = timestamp_data["WORD"]
    start = timestamp_data["START"]
    end = timestamp_data["END"]

    word_frames_dict = {}
    frames = []
    for word, start, end in zip(words, start, end):
        
        frames = []
        frame_num = math.ceil(start * video.get_meta_data()['fps'])
        while frame_num < math.floor(end * video.get_meta_data()['fps']):
            frame = video.get_data(frame_num)
            gray_frame = np.dot(frame[:,:,:3], [0.2989, 0.5870, 0.1140])
            frames.append(gray_frame)
            frame_num += 1
        
        word_frames_dict[word] = frames

        # out_file = output_dir+word+".mp4"
        # ffmpeg_extract_subclip(video, start, end, targetname=out_file)

    return word_frames_dict

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

    # container = av.open(video_filepath)

    # ims = [frame.to_image() for frame in container.decode(video=0)]
    # ims = [im.convert('L') for im in ims ]
    # ims = np.array([np.array(im) for im in ims])
    # print(ims.shape)
    # print(vid)
    return vid


def read_files(src):
    vid_data_pairs = []
    for file in os.listdir(src):
        pair = []
        if '.txt' in file:
            video_filepath = os.path.splitext(src+file)[0] + ".mp4"
            text_filepath = os.path.splitext(src+file)[0] + '.txt'

            first_line = None
            with open(text_filepath, 'r') as f:
                first_line = " ".join(f.readline().split()[1:])


            data = pd.read_csv(
                src + file, sep=" ", 
                usecols=["WORD", "START", "END", "ASDSCORE"],
                skiprows=3)
            # print(data)
            video = read_video(video_filepath)
            pair.append(data)
            pair.append(video)
            pair.append(first_line)

            if len(data) == 0:
                print("WHAT THE F")
        
            vid_data_pairs.append(pair)

    return vid_data_pairs

def read_file(file_path):
    data = pd.read_csv(
            file_path, sep=" ", 
            usecols=["WORD", "START", "END", "ASDSCORE"],
            skiprows=3)
    
    return data

def load_data():
    data_dir = './data/lrs2/sample/'
    # vid_file_path = data_dir + '00001.mp4'
    # data_file_path = data_dir + '00001.txt'

    # data = read_file(data_file_path)
    # preprocess(data_dir, vid_file_path, data)

    video_data_pairs = read_files(data_dir)
    # print(video_data_pairs)
    data_representations = []
    train_captions = []
    train_videos = []

    for [data, vid, sentence] in video_data_pairs:
        
        word_frame_dict = preprocess(data_dir, vid, data)
        data_representations.append(sentence, word_frame_dict)
        train_captions.append(sentence)
        train_videos.append(vid)

    return dict(
        train_captions = train_captions,
        train_videos = train_videos,
        data_representations = data_representations,
        
    )



def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')

if __name__ == '__main__':
    data_dir = './data/lrs2/sample/'
    create_pickle(load_data(data_dir))