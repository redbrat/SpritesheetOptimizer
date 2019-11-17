from os.path import isfile, join, splitext, basename
from os import listdir
import numpy as np
import progressbar
from numba import jit

# Подготовка массивов


def isNpArrayFile(path):
    name, extension = splitext(basename(path))
    return extension == ".npy" and name != "sizings"


allInfoFiles = [join("info",
                     f) for f in listdir("info") if isfile(join("info", f))]
allNpies = list(filter(isNpArrayFile, allInfoFiles))
allNdArrays = [np.load(path) for path in allNpies]
# for arr in allNdArrays:
#    print(arr.shape)

allLists = [numpyArr.tolist() for numpyArr in allNdArrays]
# print(allLists)
# for l in allLists:
#    print(str(len(l)) + "," + str(len(l[0])) + "," + str(len(l[0][0])))

sizings = np.load("info\\sizings.npy")
sizingsList = sizings.tolist()


# Непосредственно работа с масиивами

def countArraysOpaquePixels(ndArraysList):
    result = 0
    for ndarr in ndArraysList:
        alphas = ndarr[:, :, 3:4]
        opaque = alphas > 0
        result += opaque.sum()
    return result


def countListsOpaquePixels(l):
    result = 0
    for currentList in l:
        for x in currentList:
            for y in x:
                if (y[3] > 0):
                    result = result + 1
    return result


def twoDList(value, a, b):
    lst = [[value for col in range(b)] for col in range(a)]
    return lst


def threeDList(value, a, b, c):
    lst = [[[value for col in range(c)] for col in range(b)]
           for row in range(a)]
    return lst


def initializeCountsFromLists(sizingsArg, allListsArg):
    result = []
    for s in range(len(sizingsArg)):
        sizingX = sizingsArg[s][0]
        sizingY = sizingsArg[s][1]
        # print("size: " + str(sizingX) + "," + str(sizingY))
        result.append([])
        for l in range(len(allListsArg)):
            width = len(allListsArg[l]) - sizingX
            height = len(allListsArg[l][0]) - sizingY
            #print(str(len(allListsArg[l])) + "," + str(len(allListsArg[l][0])))
            #print(str(width) + "," + str(height))
            result[s].append(twoDList(0, width, height))
    return result


def initializeCountsFromArrays(sizings, allArrays):
    result = []
    for s in range(len(sizings)):
        sizingX = sizings[s][0]
        sizingY = sizings[s][1]
        # print("size: " + str(sizingX) + "," + str(sizingY))
        result.append([])
        shape = allArrays.shape
        for l in range(shape[0]):
            width = shape[1] - sizingX
            height = shape[2] - sizingY
            #print(str(len(allListsArg[l])) + "," + str(len(allListsArg[l][0])))
            #print(str(width) + "," + str(height))
            result[s].append(twoDList(0, width, height))
    return result


# Сначала считаем кол-во непрозрачных пикселей
unprocessedPixels = countListsOpaquePixels(allLists)
initialPixelsUnprocessed = unprocessedPixels


@jit(nopython=True)
def countSample(sampleL, sampleX, sampleY, sampleWidth, sampleHeight, allLists):
    result = 0
    repeat = False
    for l in range(len(allLists)):
        currentSprite = allLists[l]
        xDomain = len(currentSprite) - sampleWidth
        yDomain = len(currentSprite[0]) - sampleHeight
        for x in range(xDomain):
            for y in range(yDomain):
                maybeThis = True
                for xx in range(sampleWidth):
                    for yy in range(sampleHeight):
                        sampleXX = sampleX + xx
                        sampleYY = sampleY + yy
                        samplePixel = allLists[sampleL][sampleXX][sampleYY]

                        candidateXX = x + xx
                        candidateYY = y + yy
                        candidatePixel = allLists[l][candidateXX][candidateYY]

                        if (samplePixel[3] == 0 and candidatePixel[3] == 0):
                            continue

                        if (samplePixel[0] != candidatePixel[0] or samplePixel[1] != candidatePixel[1] or samplePixel[2] != candidatePixel[2] or samplePixel[3] != candidatePixel[3]):
                            maybeThis = False
                            break
                    if (maybeThis == False):
                        break

                if (maybeThis):
                    if (l < sampleL or (l == sampleL and x < sampleX) or (l == sampleL and x == sampleX and y < sampleY)):
                        repeat = True
                        break
                    else:
                        result = result + 1
            if (repeat):
                break

        if (repeat):
            break

    return result


with progressbar.ProgressBar(max_value=len(sizingsList)) as bar:
    # while unprocessedPixels > 0:
    # print("sampleCounts shape: " + str(len(sampleCounts)) + "," +
    #      str(len(sampleCounts[0])) + "," + str(len(sampleCounts[0][0])) + "," + str(len(sampleCounts[0][0][0])))
    sampleCounts = initializeCountsFromLists(sizingsList, allLists)
    for s in range(len(sizingsList)):
        sizingX = sizingsList[s][0]
        sizingY = sizingsList[s][1]
        for l in range(len(allLists)):
            sampleSprite = allLists[l]
            width = len(sampleSprite)
            height = len(sampleSprite[0])
            sampleDomainX = width - sizingX
            sampleDomainY = height - sizingY

            #print("width = " + str(width))
            #print("height = " + str(height))
            #print("sizingX = " + str(sizingX))
            #print("sizingY = " + str(sizingY))
            for sampleX in range(sampleDomainX):
                for sampleY in range(sampleDomainY):
                    #print("sampleX = " + str(sampleX))
                    #print("sampleY = " + str(sampleY))
                    #print("l = " + str(l))
                    #print("s = " + str(s))
                    sampleCounts[s][l][sampleX][sampleY] = countSample(
                        l, sampleX, sampleY, sizingX, sizingY, allLists)
        bar.update(s)

    #unprocessedPixels = countListsOpaquePixels(allLists)
    #bar.update(initialPixelsUnprocessed - unprocessedPixels)
