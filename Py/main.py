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
    for a in range(len(allArrays)):
        shape = allArrays[a].shape
        result.append(np.full((sizings.shape[0], shape[0], shape[1]), 0))
    return result


# Сначала считаем кол-во непрозрачных пикселей
unprocessedPixels = countListsOpaquePixels(allLists)
initialPixelsUnprocessed = unprocessedPixels


@jit(nopython=True)
def countSampleList(sampleL, sampleX, sampleY, sampleWidth, sampleHeight, allLists):
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


@jit(nopython=True, parallel=True)
def countSampleOnArray(sampleNdArr, sampleA, sampleX, sampleY, sampleWidth, sampleHeight, candidateNdArray, candidateA, sizings):
    result = 0
    repeat = False

    sizingsCount = sizings.shape[0]
    shape = candidateNdArray.shape
    width = shape[0]
    height = shape[1]
    xDomain = width - sampleWidth
    yDomain = height - sampleHeight

    for s in range(sizingsCount):
        for x in range(xDomain):
            for y in range(yDomain):
                maybeThis = True
                for xx in range(sampleWidth):
                    for yy in range(sampleHeight):
                        sampleXX = sampleX + xx
                        sampleYY = sampleY + yy
                        samplePixel = sampleNdArr[sampleXX][sampleYY]

                        candidateXX = x + xx
                        candidateYY = y + yy
                        candidatePixel = candidateNdArray[candidateXX][candidateYY]

                        if (samplePixel[3] == 0 and candidatePixel[3] == 0):
                            continue

                        if (samplePixel[0] != candidatePixel[0] or samplePixel[1] != candidatePixel[1] or samplePixel[2] != candidatePixel[2] or samplePixel[3] != candidatePixel[3]):
                            maybeThis = False
                            break

                    if (maybeThis == False):
                        break

                if (maybeThis):
                    if (candidateA < sampleA or (candidateA == sampleA and x < sampleX) or (candidateA == sampleA and x == sampleX and y < sampleY)):
                        repeat = True
                        result = 0
                        break
                    else:
                        result = result + 1
            if (repeat == True):
                break
        if (repeat == True):
            break
    return result

    # while unprocessedPixels > 0:
    # print("sampleCounts shape: " + str(len(sampleCounts)) + "," +
    #      str(len(sampleCounts[0])) + "," + str(len(sampleCounts[0][0])) + "," + str(len(sampleCounts[0][0][0])))


with progressbar.ProgressBar(max_value=len(allNdArrays)) as bar:
    sampleCounts = initializeCountsFromArrays(sizings, allNdArrays)
    arraysLen = len(allNdArrays)
    rangeOfAllNdArrays = range(arraysLen)
    for sampleA in rangeOfAllNdArrays:
        sampleNdArr = allNdArrays[sampleA]
        shape = sampleNdArr.shape
        width = shape[0]
        height = shape[1]
        sizingsLen = sizings.shape[0]
        for s in range(sizingsLen):
            sizingX = sizings[s, 0]
            sizingY = sizings[s, 1]
            sampleDomainX = width - sizingX
            sampleDomainY = height - sizingY
            # print("shape = " + str(shape) + ". sizingX: " + str(sizingX) + ", sizingY: " + str(sizingY) + ", sampleDomainX: " +
            #      str(sampleDomainX) + ", sampleDomainY: " + str(sampleDomainY))
            for sampleX in range(sampleDomainX):
                for sampleY in range(sampleDomainY):
                    counts = np.zeros(arraysLen)
                    for candidateA in rangeOfAllNdArrays:
                        counts[candidateA] = countSampleOnArray(
                            sampleNdArr, sampleA, sampleX, sampleY, sizingX, sizingY, allNdArrays[candidateA], candidateA, sizings)
                    # print(str(sampleA))
                    # print(sampleCounts[sampleA].shape)
                    sampleCounts[sampleA][s][sampleX][sampleY] = counts.sum()
        bar.update(sampleA)

    #unprocessedPixels = countListsOpaquePixels(allLists)
    #bar.update(initialPixelsUnprocessed - unprocessedPixels)
