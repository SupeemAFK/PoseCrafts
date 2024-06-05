import matplotlib.pyplot as plt
import numpy as np

def plot_fromPerson(person, person_idx):
        keypoints = person
        keypoints = np.array(keypoints).reshape(-1, 2)

        # Plot keypoints
        plt.scatter(keypoints[:, 0], keypoints[:, 1], s=10, c='r')

        # Connect keypoints
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
                     (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]:
            plt.plot([keypoints[i, 0], keypoints[j, 0]],
                     [keypoints[i, 1], keypoints[j, 1]], 'r')

        # Add label for each person
        plt.text(keypoints[0, 0], keypoints[0, 1], f'Person {person_idx}', fontsize=10, color='blue')

def plot_openpose(people):
    plt.figure(figsize=(8, 8))
    plt.imshow(np.zeros((300, 900, 3)))  # Create an empty image to plot keypoints on

    for idx, person in enumerate(people):
      plot_fromPerson(person, idx)

    plt.gca()  # Invert y-axis to match image coordinate system
    plt.show()