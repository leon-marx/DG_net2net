import numpy as np
import os
from PIL import Image
import torchvision.transforms as T

train_test_split = 0.8

data = []

for domain_dir in os.listdir("data/imagenet"):
    print(domain_dir)
    for content_dir in os.listdir("data/imagenet/" + domain_dir):
        print("   " + content_dir)
        for im in os.listdir("data/imagenet/" + domain_dir + "/" + content_dir):
            data.append([domain_dir, content_dir, im])

data = np.array(data)
np.random.shuffle(data)
N = data.shape[0]

train_data = data[:int(N * train_test_split)]
test_data = data[int(N * train_test_split):]

if not os.path.exists("data/imagenet_train/meta"):
    os.makedirs("data/imagenet_train/meta")
for d in train_data:
    with Image.open("data/imagenet/" + d[0] + "/" + d[1] + "/" + d[2]) as image:
        resized = T.functional.pad(image, padding=(14, 14, 15, 15), fill=256)
        if not os.path.exists("data/imagenet_train/" + d[0] + "/" + d[1]):
            os.makedirs("data/imagenet_train/" + d[0] + "/" + d[1])
        # resized.save("data/imagenet_train/" + d[0] + "/" + d[1] + "/" + d[2])
        image.save("data/imagenet_train/" + d[0] + "/" + d[1] + "/" + d[2])
    with open(f"data/imagenet_train/meta/{d[0]}.txt", "a") as f:
        f.write(d[1] + "/" + d[2] + "\n")

if not os.path.exists("data/imagenet_test/meta"):
    os.makedirs("data/imagenet_test/meta")
for d in test_data:
    with Image.open("data/imagenet/" + d[0] + "/" + d[1] + "/" + d[2]) as image:
        resized = T.functional.pad(image, padding=(14, 14, 15, 15), fill=256)
        if not os.path.exists("data/imagenet_test/" + d[0] + "/" + d[1]):
            os.makedirs("data/imagenet_test/" + d[0] + "/" + d[1])
        # resized.save("data/imagenet_test/" + d[0] + "/" + d[1] + "/" + d[2])
        image.save("data/imagenet_test/" + d[0] + "/" + d[1] + "/" + d[2])
    with open(f"data/imagenet_test/meta/{d[0]}.txt", "a") as f:
        f.write(d[1] + "/" + d[2] + "\n")
