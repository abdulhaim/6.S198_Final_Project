
from preprocessing.video_to_frame import convert_to_frames
from cnn.generate_cnn_output import main

from rnn.rnn_eval import main_rnn
if __name__ == "__main__":
    # convert test video to frames
    #print("Converting Test Video to Frames")
    #word_count = 10 
    #input_type = "/train/"
    #output_pickle_name = 'preprocessing/pickle_data/labeled-frames-train.pkl'
    #convert_to_frames("preprocessing/raw_data/",word_count,input_type,output_pickle_name)

    # generate predicted output 
    #print("Generating Predicted Output")
    #main("preprocessing/pickle_data/labeled-frames-train","results/train-demo",15)

    # run rnn_eval
    print("Running RNN Eval")
    filename = 'predicted-frames-2.pkl'
    input_length = 10  #64
    # input_length = 2048
    frames = 15
    batch_size = 32
    num_classes = 10 #64

    main_rnn(filename, frames, batch_size, num_classes, input_length)
                                                                   

