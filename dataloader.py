import cv2

# This file must be collocated with the lfw dataset directory

def read_pairs_file(f):
    people_matches = {}
    num_pairs = int(f.readline())
    loadedSet = []

    for pair in range(num_pairs):
        words = f.readline().strip().split()
        picA = words[0] + '_' + words[1].zfill(4)
        picB = words[0] + '_' + words[2].zfill(4)
        loadedSet.append([picA, picB, True])

    for pair in range(num_pairs):
        words = f.readline().strip().split()
        picA = words[0] + '_' + words[1].zfill(4)
        picB = words[2] + '_' + words[3].zfill(4)
        loadedSet.append([picA, picB, False])

    return loadedSet

def load_train_test():
    test_set = []
    train_set = []
    with open('pairsDevTrain.txt') as f:
        train_set = read_pairs_file(f)
    with open('pairsDevTest.txt') as f:
        test_set = read_pairs_file(f)
    return {'train' : train_set, 'test' : test_set}

def load_image(name_num):
    name = name_num[:-5]
    return cv2.imread('./lfw/'+name+ '/' + name_num +'.jpg')
