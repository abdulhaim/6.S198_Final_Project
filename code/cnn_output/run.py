import os

l=os.listdir("/home/abdulhai/6.S198_Final_Project/code/majorData/")

def dotremove(poi):  
	for x in poi:
		if x.startswith('.'):
			poi.remove(x)
dotremove(l) 
for x in l:
	txt=str(x)+'.sh'
	file=open(txt,"w")
	p=os.listdir("/home/abdulhai/6.S198_Final_Project/code/majorData/"+x)
	dotremove(p)
	for j in p:
		with open(txt,"a") as text_file:
			text = "python label_image.py --image=/home/abdulhai/6.S198_Final_Project/code/majorData/Accept/050_001_001_frame_0.jpeg --graph=/home/abdulhai/6.S198_Final_Project/code/tmp/retrained_graph.pb --labels=/home/abdulhai/6.S198_Final_Project/code/tmp/retrained_labels.txt --input_layer=\"Placeholder\" --output_layer=\"final_result\"" + "\n"			
			text_file.write(text)
