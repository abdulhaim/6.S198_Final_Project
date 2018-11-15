"""
Run a holdout set of data through our trained RNN. Requires we first
run train_rnn.py and save the weights.
"""
from rnn_utils import get_network, get_network_deep, get_network_wide, get_data
import tflearn
import numpy as np


def main_rnn(filename, frames, batch_size, num_classes, input_length):
    """From the blog post linked above."""
    # Get our data.
    X_train, y_train = get_data(
        filename, frames, num_classes, input_length, False)
   
    # Get sizes.
    print("Y train :- ", y_train[0])
    num_classes = len(y_train[0])

    # Get our network.
    net = get_network_wide(frames, input_length, num_classes)

    # Get our model.
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.load('checkpoints/rnn.tflearn')
    #model.load('checkpoints_backup/rnn.tflearn')

    # Evaluate.
    hc = model.predict(X_train)
    hc = [np.argmax(every) for every in hc]
    aadi = [np.argmax(every) for every in y_train]
    answer = []

    for i in range(0, len(hc)):
	 answer.append([aadi[i], hc[i]])

    answer.sort()
    f = open("results_demo/results.txt", "wb")
    for x in answer:
        print(x[0], x[1])
    	f.write(str(x[0])+" "+str(x[1])+"\n")

    print("Evaluation of model:", model.evaluate(X_train, y_train))

if __name__ == '__main__':
    filename = 'results/predicted-frames-2.pkl'
    input_length = 10 #64
    # input_length = 2048
    frames = 15 #201
    batch_size = 32
    num_classes = 10 #64

    main_rnn(filename, frames, batch_size, num_classes, input_length)
