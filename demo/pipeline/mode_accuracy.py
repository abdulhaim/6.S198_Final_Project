import shutil
import random
import os
from scripts.label_image import *

#dictionary to hold the number of times a word was predicted, uncomment the dic for the model you plan on using: 10 words or 64 words
word_count_dic = {"00": 0, "01": 0, "02": 0, "03": 0, "04": 0, "05": 0, "06": 0, "07": 0, "08": 0, "09": 0}
name_dic = {"00": "Accept", "01": "Appear", "02": "Argentina", "03": "Away", "04": "Barbeque", "05": "Bathe", "06": "Birthday", "07": "Bitter", "08": "Born", "09": "Breakfast"}
#word_count_dic = {'accept': 0, 'appear': 0, 'argentina': 0, 'away': 0, 'barbecue': 0, 'bathe': 0, 'birthday': 0, 'bitter': 0, 'born': 0, 'breakfast': 0, 'bright': 0, 'buy': 0, 'call': 0, 'candy': 0, 'catch': 0, 'chewing gum': 0, 'coin': 0, 'colors': 0, 'copy': 0, 'country': 0, 'dance': 0, 'deaf': 0, 'drawer': 0, 'enemy': 0, 'find': 0, 'food': 0, 'give': 0, 'green': 0, 'help': 0, 'hungry': 0, 'last name': 0, 'learn': 0, 'light blue': 0, 'man': 0, 'map': 0, 'milk': 0, 'mock': 0, 'music': 0, 'name': 0, 'none': 0, 'opaque': 0, 'patience': 0, 'perfume': 0, 'photo': 0, 'realize': 0, 'red': 0, 'rice': 0, 'run': 0, 'ship': 0, 'shut down': 0, 'skimmer': 0, 'son': 0, 'spaghetti': 0, 'sweet milk': 0, 'thanks': 0, 'to land': 0, 'trap': 0, 'unknown': 0, 'uruguay': 0, 'water': 0, 'where': 0, 'women': 0, 'yellow': 0, 'yogurt': 0} 

def predict_word(directory, model_file, label_file, input_height, input_width, input_mean, input_std, input_layer, output_layer, graph, input_name, output_name, input_operation, output_operation, sample_size):
	"""
	Given a directory with segmented video frames:
	returns the most frequently predicted word
	by the model of a random sample of frames 
	of the video
	"""
	images = random.sample(os.listdir(directory), sample_size)
	#load all images instead:
	#images = os.listdir(directory)

	total_time = 0
	temp_dic =  word_count_dic.copy()

	for image in images:
		#path to image
		if image == '.DS_Store':
			continue 
		path = os.path.join(directory, image)

		t = read_tensor_from_image_file(path, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)

		with tf.Session(graph=graph) as sess:
			start = time.time()
			results = sess.run(output_operation.outputs[0],
                  {input_operation.outputs[0]: t})
			end=time.time()
			total_time += (end-start)
			results = np.squeeze(results)

		top_k = results.argsort()[-5:][::-1]
		labels = load_labels(label_file)

		#print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
		template = "{} (score={:0.5f})"
		top_index = top_k[0]

		temp_dic[labels[top_index]] += 1

	print("Word Counts: ", temp_dic)
	predicted_word = max(temp_dic.keys(), key=(lambda key: temp_dic[key]))
	print("The Predicted Word Is:", name_dic[predicted_word])
	print("The Evaluation Time Was:", total_time)
	print(' ')
	return name_dic[predicted_word]
