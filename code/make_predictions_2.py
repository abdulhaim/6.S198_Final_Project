import label_image 

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



def main():
    with open('data/labeled-frames-' + batch + '.pkl', 'rb') as fin:
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