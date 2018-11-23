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
    frame_predictions = []
    
    print(len(frames))
    prev_time = datetime.datetime.now().replace(microsecond=0)
    count = 0
    for i, frame in tqdm(enumerate(frames)):
        filename = frame[0]
        label = frame[1]
        frameCount = frame[2]
        image = frame[0]
	if(count%200 == 0):
		print(count)
        prediction = label_image.get_prediction(filename)
        #print(prediction)
	frame_predictions.append([prediction, label, frameCount])
    	count = count+1
    return frame_predictions

def get_accuracy(predictions, labels):
    """After predicting on each batch, check that batch's
    accuracy to make sure things are good to go. This is
    a simple accuracy metric, and so doesn't take confidence
    into account, which would be a better metric to use to
    compare changes in the model."""
    correct = 0
    for frame in predictions:
        # Get the highest confidence class.
        this_prediction = frame[0].tolist()
        # print this_prediction
        this_label = frame[1]
        # print this_label

        max_value = max(this_prediction)
        max_index = this_prediction.index(max_value)
        predicted_label = labels[max_index]
        # print predicted_label

        # Now see if it matches.
        print(predicted_label, this_label)
        if predicted_label.lower() == this_label.lower():
            correct += 1
        print(correct, len(predictions))

    print(correct, len(predictions))
    accuracy = correct / float(len(predictions))
    return accuracy

def takespread(sequence, num):
    length = float(len(sequence))
    for i in range(num):
        yield sequence[int(ceil(i * length / num))]

def main():
    with open('preprocessing/pickle_data/labeled-frames-2' + '.pkl', 'rb') as fin:
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
        elements = takespread(final_dict[key],15)
        new_frames.extend(elements)

    print("size:", len(new_frames))
    predictions = predict_on_frames(new_frames)

    for frame in predictions:
        print(frame)

    #accuracy = get_accuracy(predictions, labels)
    #print("Batch accuracy: %.5f" % accuracy)

    # Save it.
    with open('predicted-frames-2' + '.pkl', 'wb') as fout:
        pickle.dump(predictions, fout)


if __name__ == "__main__":
    main()

