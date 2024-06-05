import json
import os

def list_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            yield os.path.join(dirpath, filename)

folder_path = "[FOLDER_PATH]"
for file_path in list_files(folder_path):
    with open(file_path, 'r') as r:
        data = json.load(r)
        data['people'].sort(key=lambda x: x['pose_keypoints_2d'][0])

    with open(file_path, 'w') as w:
        json.dump(data, w)

print("Complete")