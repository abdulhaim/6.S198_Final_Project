import sys
import pickle
"""Viewing Contents of Pickle File"""
x = pickle.load(open("data/labeled-frames-1.pkl"))

print("Length of file: {:,}".format(len(x)))
print("Enter y to view number of files for each word and n to exit")
user_input = raw_input()
count = 0
word = ""
if(user_input == 'y'):
	for element in x:
		if element[1] != word:
			print(word + " {:,}".format(count))
			word = element[1]
			count = 0
		count+=1

    

