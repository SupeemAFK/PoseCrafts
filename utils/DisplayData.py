import os
from colorama import Fore

def count_files(directory):
    total_files = 0
    for root, dirs, files in os.walk(directory):
        total_files += len(files)
    return total_files

def display_tree(directory, indent=0):
    if not os.path.isdir(directory):
        return

    # Display current directory
    print(Fore.WHITE + "|   " * indent + "|---" + os.path.basename(directory) + ((15 - len(os.path.basename(directory))) * " "), end="")
    
    # Count files in current directory
    file_count = count_files(directory)
    print(Fore.LIGHTCYAN_EX + " ({0} Sequences)".format(file_count))

    # Display subdirectories recursively
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            display_tree(item_path, indent + 1)
