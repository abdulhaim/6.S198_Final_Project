import label_image 
import pickle 
import tensorflow as tf
import sys
from tqdm import tqdm


def predict_on_frames():
    frame_predictions = []

    for i, frame in enumerate(frames):
        filename = frame[0]
        label = frame[1]
        frameCount = frame[2]

        image = frame[0]

        prediction = label_image.main(filename)

        frame_predictions.append([prediction, label, frameCount])

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
        print correct, len(predictions)

    print(correct, len(predictions))
    accuracy = correct / float(len(predictions))
    return accuracy

def main():
    with open('data/labeled-frames-1' + '.pkl', 'rb') as fin:
        frames = pickle.load(fin)

    predictions = predict_on_frames(frames, batch)
    for frame in predictions:
        print(frame)
    accuracy = get_accuracy(predictions, labels)
    print("Batch accuracy: %.5f" % accuracy)

    # Save it.
    with open('data/predicted-frames-' + batch + '.pkl', 'wb') as fout:
        pickle.dump(predictions, fout)


if __name__ == "__main__":
    main()