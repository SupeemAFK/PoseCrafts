import os
import json

def list_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            yield os.path.join(dirpath, filename)

folder_path = '[FOLDER_PATH]'
for file_path in list_files(folder_path):
  with open(file_path) as f:
    data = json.load(f)
    if (data["canvas_width"] != 900 or data["canvas_height"] != 300 or len(data["people"]) != 5 ): print(file_path)