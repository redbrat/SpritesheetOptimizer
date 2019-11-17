from os.path import isfile, join, splitext, basename
from os import listdir
import numpy as np


def isNpArrayFile(path):
    name, extension = splitext(basename(path))
    return extension == ".npy" & name != "sizings"


allInfoFiles = [join("info",
                     f) for f in listdir("info") if isfile(join("info", f))]
allNpies = list(filter(isNpArrayFile, allInfoFiles))
allNdArrays = [np.load(path) for path in allNpies]
for arr in allNdArrays:
    print(arr.shape)

sizings = np.load("info\\sizings.npy")
