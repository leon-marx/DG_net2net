import numpy as np
import os
from PIL import Image
import torchvision.transforms as T

train_test_split = 0.8

data = []

main_dir = "/home/../hd-data/imagenet/pacs_classes_only"
train_dir = "/home/../hd-data/imagenet/imagenet_train"
test_dir = "/home/../hd-data/imagenet/imagenet_test"

for content_dir in os.listdir(main_dir):
    print(content_dir)
    for im in os.listdir(main_dir + "/" + content_dir):
        data.append(["photo", content_dir, im])

data = np.array(data)
np.random.shuffle(data)
N = data.shape[0]

train_data = data[:int(N * train_test_split)]
test_data = data[int(N * train_test_split):]

if not os.path.exists(train_dir + "/meta"):
    os.makedirs(train_dir + "/meta")
for d in train_data:
    with Image.open(main_dir + "/" + d[1] + "/" + d[2]) as image:
        if not os.path.exists(train_dir + "/photo/" + d[1]):
            os.makedirs(train_dir + "/photo/" + d[1])
        image.save(train_dir + "/photo/" + d[1] + "/" + d[2])
    with open(f"{train_dir}/meta/photo.txt", "a") as f:
        f.write(d[1] + "/" + d[2] + "\n")

if not os.path.exists(test_dir + "/meta"):
    os.makedirs(test_dir + "/meta")
for d in test_data:
    with Image.open(main_dir + "/" + d[1] + "/" + d[2]) as image:
        if not os.path.exists(test_dir + "/photo/" + d[1]):
            os.makedirs(test_dir + "/photo/" + d[1])
        image.save(test_dir + "/photo/" + d[1] + "/" + d[2])
    with open(f"{test_dir}/meta/photo.txt", "a") as f:
        f.write(d[1] + "/" + d[2] + "\n")
