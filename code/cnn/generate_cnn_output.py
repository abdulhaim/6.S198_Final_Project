import label_image 
import numpy as np
import pickle 
import tensorflow as tf
import sys
from tqdm import tqdm
import datetime
from math import ceil
import itertools
import operator

def predict_on_frames(frames):
    """
    Return a list containing the predicted output for each frame image 
    from the retrained CNN model
    """
    frame_predictions = []
    print("Total Number of Frames ",len(frames))
    count = 0
    #for i, frame in tqdm(enumerate(frames)):
    for frame in tqdm(frames):
        filename = frame[0]
        label = frame[1]
        frameCount = frame[2]

        if(count%200 == 0):
            print(count)
        
        prediction = label_image.get_prediction(filename)
        
        frame_predictions.append([prediction, label, frameCount])
        count = count + 1

    return frame_predictions

def takespread(sequence, num):
    """
    Returns list of elements with length "num" that are found to be equally spaced in the sequence provided 
    """
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]

def main(input_file_name,output_file_name,video_length):
    """
    Reads in the labelled frames and saves output from CNN model in pickle file
    """
    length_of_video = video_length
    with open(input_file_name + '.pkl', 'rb') as fin:
        frames = pickle.load(fin)

    sorted_frames = list(list(x[1]) for x in itertools.groupby(frames, operator.itemgetter(1)))
    final_dict = dict()
    for element in sorted_frames:
        for f in element:
            name = f[0]
            video_name = name[name.rindex("/")+1:name.rindex("frame")-1]
            if video_name not in final_dict:
                final_dict[video_name] = []

            final_dict[video_name].append(f)

    new_frames = []

    for key in final_dict:
        #elements = takespread(final_dict[key],length_of_video)
        new_frames.extend(final_dict[key])

    print("size:", len(new_frames))
    predictions = predict_on_frames(new_frames)

    with open(output_file_name + '.pkl', 'wb') as fout:
        pickle.dump(predictions, fout)

if __name__ == "__main__":
	main("/home/abdulhai/6.S198_Final_Project/code/pickle_data/pickle_train","/home/abdulhai/6.S198_Final_Project/code/results/predicted-frames-train.pkl",200)


