from os import listdir
from os.path import isfile, join

import os
import shutil
home_directory = os.getcwd() + "/raw_data"
files = [f for f in listdir(home_directory) if isfile(join(home_directory, f))]

name_dict = {"Opaque": "001", 
			 "Light-Red": "002", 
			 "Green": "003", 
			 "Yellow": "004",
			 "Bright": "005",
			 "Light-blue": "006",
			 "Colors": "007",
			 "Red": "008",
			 "Women": "009", 
			 "Enemy": "010",
			 "Son": "011",
			 "Man": "012",
			 "Away": "013",
			 "Drawer": "014",
			 "Born": "015",
			 "Learn": "016",
			 "Call": "017",
			 "Skimmer": "018",
			 "Bitter": "019",
			 "Sweet milk": "020",
			 "Milk": "021",
			 "Water": "022",
			 "Food": "023",
			 "Argentina": "024",
			 "Uruguay": "025",
			 "Country": "026",
			 "Last name": "027",
			 "Where":"028",
			 "Mock": "029",
			 "Birthday": "030",
			 "Breakfast": "031",
			 "Photo": "032",
			 "Hungry": "033",
			 "Map": "034",
			 "Coin": "035",
			 "Music": "036",
			 "Ship": "037",
			 "None": "038",
			 "Name": "039",
			 "Patience": "040",
			 "Perfume": "041",
			 "Deaf": "042",
			 "Trap": "043",
			 "Rice": "044",
			 "Barbecue": "045",
			 "Candy": "046",
			 "Chewing-gum": "047",
			 "Spaghetti": "048",
			 "Yogurt": "049",
			 "Accept": "050",
			 "Thanks": "051",
			 "Shut down": "052",
			 "Appear": "053",
			 "To land": "054",
			 "Catch": "055",
			 "Help": "056",
			 "Dance": "057",
			 "Bathe": "058",
			 "Buy": "059",
			 "Copy": "060",
			 "Run": "061",
			 "Realize": "062",
			 "Give": "063",
			 "Find": "064"}

for name, number in name_dict.items():
	output_names = [f for f in files if (f[0:3] == number)]
	print("Home Directory",home_directory)
	print("Name",name)
	for file_name in output_names: 
		if not os.path.exists(home_directory + "/" +  name):
			os.mkdir(home_directory + "/" + name)
		current_directory = home_directory + "/" + file_name
		print(current_directory)
		new_directory = home_directory + "/" +  name + "/" + file_name
		shutil.move(current_directory, new_directory)
		print(name, " Moved!")

