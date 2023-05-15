import os, sys, tarfile, errno
import numpy as np
import matplotlib.pyplot as plt

import urllib.request
from imageio import imsave
from tqdm import tqdm
import random
import shutil

''' The following code was found at https://github.com/mttk/STL10/blob/master/stl10_input.py'''

#define global variables
HEIGHT = 96
WIDTH = 96
DEPTH = 3

SIZE = HEIGHT * WIDTH * DEPTH

DATA_DIR = 'Data'
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

TRAIN_DATA_PATH = 'Data/stl10_binary/train_X.bin'
TRAIN_LABEL_PATH = 'Data/stl10_binary/train_y.bin'

TEST_DATA_PATH = 'Data/stl10_binary/test_X.bin'
TEST_LABEL_PATH = 'Data/stl10_binary/test_y.bin'


def download_and_extract():
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main():
    # download and unzip
    download_and_extract()
    # get train images
    train_labels = read_labels(TRAIN_LABEL_PATH)
    train_images = read_all_images(TRAIN_DATA_PATH)
    # get test images
    test_labels = read_labels(TEST_LABEL_PATH)
    test_images = read_all_images(TEST_DATA_PATH)
    # save images
    save_images(train_images, train_labels, "train")
    save_images(test_images, test_labels, "test")


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):

    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)

        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def save_images(images, labels, types):
    i = 0
    for image in tqdm(images, position=0):
        label = labels[i] 
        directory = DATA_DIR + '/' + types + '/' + str(label) + '/'
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        imsave("%s.png" % filename, image, format="png")
        i = i+1


if __name__ == '__main__':
    main()
