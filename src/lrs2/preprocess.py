import tensorflow as tf
import numpy as np
import os
import pandas as pd
import imageio
# from pylab import *
import pylab

def preprocess(video, timestamp_data):
    print(timestamp_data)
    words = timestamp_data["WORD"]
    start = words = timestamp_data["START"]
    end = timestamp_data["END"]

    word_frames_dict = {}
    for num, image in enumerate(video.iter_data()):
        timestamp = float(num)/ video.get_meta_data()['fps']
        print(timestamp)

def read_videos(src):
    videos = []
    for file in os.listdir(src):
        if '.mp4' in file:
            vid = imageio.get_reader(src+file, 'ffmpeg')
            videos.append(vid)
    
    return videos


def read_files(src):
    files = []
    for file in os.listdir(src):
        if '.txt' in file:
            data = pd.read_csv(
                src + file, sep=" ", 
                usecols=["WORD", "START", "END", "ASDSCORE"],
                skiprows=3)
            print(data)
            files.append(data)

    return files

def main():
    data_dir = './data/lrs2/sample/'
    data_files = read_files(data_dir)
    vids = read_videos(data_dir)
    video_data_pairs = zip(os.listdir(data_dir), vids, data_files,)
    for filename, vid, data in video_data_pairs:
        print(filename)
        preprocess(vid, data)
        # image = vid.get_data(10)
        # fig = pylab.figure()
        # fig.suptitle('image #{}'.format(10), fontsize=20)
        # pylab.imshow(image)
        # pylab.show()
        # print(data)
    # preprocess(data_files)


if __name__ == '__main__':
    main()