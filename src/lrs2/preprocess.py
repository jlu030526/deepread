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
            frames += video.get_data(frame_num)
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

def read_videos(src):
    videos = []
    for file in os.listdir(src):
        if '.mp4' in file:
            videos.append(src+file)
            # vid = imageio.get_reader(src+file, 'ffmpeg')
            # videos.append(vid)
            # read_audios(src+file)
            # wave.open('ffaudio.wav', 'rb')
    
    return videos


def read_files(src):
    files = []
    for file in os.listdir(src):
        if '.txt' in file:
            data = pd.read_csv(
                src + file, sep=" ", 
                usecols=["WORD", "START", "END", "ASDSCORE"],
                skiprows=3)
            # print(data)
            files.append(data)

    return files

def read_file(file_path):
    data = pd.read_csv(
            file_path, sep=" ", 
            usecols=["WORD", "START", "END", "ASDSCORE"],
            skiprows=3)
    
    return data

def main():
    data_dir = './data/lrs2/sample/'
    vid_file_path = data_dir + '00001.mp4'
    data_file_path = data_dir + '00001.txt'

    data = read_file(data_file_path)
    preprocess(data_dir, vid_file_path, data)

    # data_files = read_files(data_dir)
    # vids = read_videos(data_dir)
    # video_data_pairs = zip(os.listdir(data_dir), vids, data_files)
    # for filename, vid, data in video_data_pairs:
    #     print(filename)
    #     word_frame_dict = preprocess(data_dir, vid, data)
        
        # image = vid.get_data(10)
        # fig = pylab.figure()
        # fig.suptitle('image #{}'.format(10), fontsize=20)
        # pylab.imshow(image)
        # pylab.show()
        # print(data)
    # preprocess(data_files)


if __name__ == '__main__':
    main()