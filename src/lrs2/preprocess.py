import tensorflow as tf
import numpy as np
import os
import pandas as pd
import imageio
# from pylab import *
import pylab
import wave
import moviepy
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def preprocess(output_dir, video, timestamp_data):
    # print(timestamp_data)
    words = timestamp_data["WORD"]
    start = timestamp_data["START"]
    end = timestamp_data["END"]

    word_frames_dict = {}
    frames = []
    for word, start, end in zip(words, start, end):
        
        frames = []
        frame_num = int(start * video.get_meta_data()['fps'])
        while frame_num < int(end * video.get_meta_data()['fps']):
            frames.append(video.get_data(frame_num))
            frame_num += video.get_meta_data()['fps']
        
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

    return vid


def read_files(src):
    vid_data_pairs = []
    for file in os.listdir(src):
        pair = []
        if '.txt' in file:
            video_filepath = os.path.splitext(src+file)[0] + ".mp4"
            data = pd.read_csv(
                src + file, sep=" ", 
                usecols=["WORD", "START", "END", "ASDSCORE"],
                skiprows=3)
            # print(data)
            video = read_video(video_filepath)
            pair.append(data)
            pair.append(video)

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

def main():
    data_dir = './data/lrs2/sample/'
    # vid_file_path = data_dir + '00001.mp4'
    # data_file_path = data_dir + '00001.txt'

    # data = read_file(data_file_path)
    # preprocess(data_dir, vid_file_path, data)

    video_data_pairs = read_files(data_dir)
    # print(video_data_pairs)

    for [data, vid] in video_data_pairs:
        
        # print(filename)
        # print(data)
        word_frame_dict = preprocess(data_dir, vid, data)
        # print(word_frame_dict)
        
    #     image = vid.get_data(10)
    #     fig = pylab.figure()
    #     fig.suptitle('image #{}'.format(10), fontsize=20)
    #     pylab.imshow(image)
    #     pylab.show()
    #     print(data)
    # preprocess(data_files)


if __name__ == '__main__':
    main()