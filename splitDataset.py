import os, shutil
import random


def copyfile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


def splitDataset(datasetDirs, trainDir, testDir, k):
    train = []
    test = []
    label = 0
    for datasetDir in datasetDirs:
        if not os.path.isdir(trainDir):
            os.mkdir(trainDir)
        if not os.path.isdir(testDir):
            os.mkdir(testDir)
        for dir in os.listdir(datasetDir):
            files = []
            dir = os.path.join(datasetDir, dir)
            for i, name in enumerate(os.listdir(dir)):
                files.append((os.path.join(dir, name), str(label) + "_" + str(i) + ".jpg"))
            random.shuffle(files)
            train += files[k:]
            test += files[:k]
            label += 1
    for src, trgName in train:
        copyfile(src, os.path.join(trainDir, trgName))
    for src, trgName in test:
        copyfile(src, os.path.join(testDir, trgName))


if __name__ == "__main__":
    splitDataset(["./data/faces94", "./data/faces95", "./data/faces96", "./data/grimace"],
                 "./data/train", "./data/test", 5)
