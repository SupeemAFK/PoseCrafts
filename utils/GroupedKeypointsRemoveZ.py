import json

def groupedKeypointsForPerson(person):
  pose_keypoints_2d = person["pose_keypoints_2d"]
  grouped_keypoints = []

  for i in range(len(pose_keypoints_2d)):
    if (i+1) % 3 == 0:
      grouped_keypoints.append([pose_keypoints_2d[i-2], pose_keypoints_2d[i-1]])

  return grouped_keypoints

def groupedKeypointForJSON(json_path):
  newData = []
  with open(json_path) as f:
    data = json.load(f)
    for i, person in enumerate(data['people']):
      grouped_keypoints = groupedKeypointsForPerson(person)
      newData.append(grouped_keypoints)
  return newData