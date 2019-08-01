import os
cwd = os.getcwd() # get current work directory
rootdir = cwd

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file == '.DS_Store':
            path = os.path.join(subdir, file)
            os.remove(path)

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(os.path.join(subdir, file))