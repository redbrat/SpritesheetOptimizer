from os.path import isfile, join, splitext
from os import listdir
import numpy as np


def isNpArrayFile(path):
    _, file_extension = splitext(path)
    return file_extension == ".npy"


allInfoFiles = [join("info",
                     f) for f in listdir("info") if isfile(join("info", f))]
allNpies = list(filter(isNpArrayFile, allInfoFiles))
allNdArrays = [np.load(path) for path in allNpies]
for arr in allNdArrays:
    print(arr.shape)
