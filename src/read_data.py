import json
import os
import numpy as np
import pandas as pd
import glob
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img


def read_data(path_to_dataset: str = "data/dataset/",) -> [str, int]:
    labels = list(map(os.path.basename, glob.glob(path_to_dataset+'*')))
    number_of_classes = len(labels)
    return labels, number_of_classes

def prepare_dataset(labels, path_to_dataset: str = "data/dataset/", path_to_config: str = "read_data_config.json") -> [np.array, np.array]:
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    dataset = []
    count = 0
    for label in labels:
        folder = os.path.join(path_to_dataset,label)
        files_in_folder = os.listdir(folder)
        for image in files_in_folder:
            img=load_img(os.path.join(folder,image), target_size=config['image_size'])
            np_img=img_to_array(img)
            np_img=np_img/255.0
            dataset.append((np_img,count))
        count+=1
    random.shuffle(dataset)
    x, y = zip(*dataset)

    return np.array(x), np.array(y)