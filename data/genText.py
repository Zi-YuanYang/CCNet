import os
import numpy as np

path1 = '../TongJi/session1/'
path2 = '../TongJi/session2/'


root = './'


with open(os.path.join(root, 'train_Tongji.txt'), 'w') as ofs:
    files = os.listdir(path1)
    files.sort()
    for filename in files:

        userID = int(filename[:5])
        sampleID = userID % 10
        userID = int((userID-1)/10)
        imagePath = os.path.join(path1, filename)
        # if sampleID == ID_number:
        ofs.write('%s %d\n'%(imagePath, userID))
            # ID_number = 0

with open(os.path.join(root, 'test_right.txt'), 'w') as ofs:
    files = os.listdir(path2)
    files.sort()
    for filename in files:
        # print(filename)
        userID = int(filename[:5])
        userID = int((userID-1)/10)
        # print(userID)
        imagePath = os.path.join(path2, filename)
        # if userID % 2 != 0:
        ofs.write('%s %d\n'%(imagePath,userID))
