from os.path import isfile, join, splitext
from os import listdir
import numpy as np


def isNpArrayFile(path):
    filename, file_extension = splitext(path)
    return file_extension == ".npy"


allNpies = list(
    filter(isNpArrayFile, [f for f in listdir(".") if isfile(join(".", f))]))
allNdArrays = [np.load(path) for path in allNpies]
for arr in allNdArrays:
    print(arr.shape)
