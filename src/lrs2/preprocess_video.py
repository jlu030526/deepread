from preprocess import read_files, preprocess
import numpy as np
import cv2
import av

def test():
    data_dir = './data/lrs2/sample/'
    video_data_pairs = read_files(data_dir)

    for [data, vid] in video_data_pairs:
        
        # print(filename)
        # print(data)
        word_frame_dict = preprocess(data_dir, vid, data)

        for word in word_frame_dict.keys():
            frames = word_frame_dict[word]

            size = 160,160
            
            fps = 25
            # duration = len(frames)/fps
            out = cv2.VideoWriter(f'output{word}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
            for i in range(len(frames)):
                # print(i)
                data = frames[i].astype('uint8')
                data.reshape(size)
                # print(frames[i])
                # data = np.random.randint(0, 256, size, dtype='uint8')
                out.write(data)
            out.release()
            
    

test()