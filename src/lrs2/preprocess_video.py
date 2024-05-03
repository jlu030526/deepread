from preprocess import read_files, preprocess
import numpy as np
import cv2
import av

def test():
    data_dir = './data/lrs2/sample/'
    video_data_pairs = read_files(data_dir)
    # print((video_data_pairs[0]))
    # print((video_data_pairs[1]))
    # print((video_data_pairs[2]))

    for [data, vid, captions] in video_data_pairs:
        
        # print(filename)
        # print(data)
        word_frame_dict = preprocess(data_dir, vid, data)

        for word in word_frame_dict.keys():
            frames = word_frame_dict[word]
            # gray_frames = np.dot(frames[:,:,:3], [0.2989, 0.5870, 0.1140])

            size = 160,160
            
            fps = 25
            # duration = len(frames)/fps
            out = cv2.VideoWriter(f'output{word}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
            for i in range(len(frames)):
                # print(i)
                gray_frame = np.dot(frames[i][:,:,:3], [0.2989, 0.5870, 0.1140])
                data = gray_frame.astype('uint8')
                data.reshape(size)
                # print(frames[i])
                # data = np.random.randint(0, 256, size, dtype='uint8')
                out.write(data)
            out.release()
            
    

test()