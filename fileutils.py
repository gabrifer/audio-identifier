import os 

def find_file_by_name(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)