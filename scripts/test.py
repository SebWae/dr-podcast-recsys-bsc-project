import os
import pandas as pd
import sys

# adding the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

import json

# Load the JSON file
with open("utils/utils.json", "r") as file:
    data = json.load(file)
show_episode_dict = data["show_episodes_dict"]

print(show_episode_dict["Genvej"])