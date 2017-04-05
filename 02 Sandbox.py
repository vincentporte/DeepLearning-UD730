#Sandbox.py
import numpy as np

a = 1000000000
b = 0.000001
c = a

for x in xrange(1000000):
    c = c + b
    
c = c - a
print c

########################################################################
import os
from IPython.display import Image, display

listOfImageNames = ['notMNIST_large/I/a2FkZW4udHRm.png',
                    'notMNIST_large/J/aG9vZ2UgMDRfNjUudHRm.png',
                    'notMNIST_large/A/aG9tZXdvcmsgbm9ybWFsLnR0Zg==.png',
                    'notMNIST_large/F/a2FuIEUudHRm.png']

print (os.path.abspath(''))

for imageName in listOfImageNames:
    display(Image(filename=imageName))
########################################################################
import os
print (os.path.abspath(''))

root = 'notMNIST_large'
train_folders = [os.path.join(root,o) for o in sorted (os.listdir(root)) if os.path.isdir(os.path.join(root,o))]
print train_folders
for tf in train_folders:
    print tf, len([name for name in os.listdir(tf) if os.path.isfile(os.path.join(tf, name))])

root = 'notMNIST_small'
test_folders = [os.path.join(root,o) for o in sorted (os.listdir(root)) if os.path.isdir(os.path.join(root,o))]
print test_folders
for tf in test_folders:
    print tf, len([name for name in os.listdir(tf) if os.path.isfile(os.path.join(tf, name))])

>> resultats
['notMNIST_large\\A', 'notMNIST_large\\B', 'notMNIST_large\\C', 'notMNIST_large\\D', 'notMNIST_large\\E', 'notMNIST_large\\F', 'notMNIST_large\\G', 'notMNIST_large\\H', 'notMNIST_large\\I', 'notMNIST_large\\J']
notMNIST_large\A 52912
notMNIST_large\B 52912
notMNIST_large\C 52881
notMNIST_large\D 52912
notMNIST_large\E 52912
notMNIST_large\F 52912
notMNIST_large\G 52912
notMNIST_large\H 52912
notMNIST_large\I 52912
notMNIST_large\J 52911
['notMNIST_small\\A', 'notMNIST_small\\B', 'notMNIST_small\\C', 'notMNIST_small\\D', 'notMNIST_small\\E', 'notMNIST_small\\F', 'notMNIST_small\\G', 'notMNIST_small\\H', 'notMNIST_small\\I', 'notMNIST_small\\J']
notMNIST_small\A 1873
notMNIST_small\B 1873
notMNIST_small\C 1873
notMNIST_small\D 1873
notMNIST_small\E 1873
notMNIST_small\F 1873
notMNIST_small\G 1872
notMNIST_small\H 1872
notMNIST_small\I 1872
notMNIST_small\J 1872

########################################################################
