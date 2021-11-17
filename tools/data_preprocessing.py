import numpy as np
import os
from PIL import Image
import torchvision.transforms as T

train_test_split = 0.8

data = []

for domain_dir in os.listdir("data/PACS"):
    print(domain_dir)
    for content_dir in os.listdir("data/PACS/" + domain_dir):
        print("   " + content_dir)
        for im in os.listdir("data/PACS/" + domain_dir + "/" + content_dir):
            data.append([domain_dir, content_dir, im])

data = np.array(data)
np.random.shuffle(data)
N = data.shape[0]

train_data = data[:int(N * train_test_split)]
test_data = data[int(N * train_test_split):]

if not os.path.exists("data/PACS_train/meta"):
    os.makedirs("data/PACS_train/meta")
for d in train_data:
    with Image.open("data/PACS/" + d[0] + "/" + d[1] + "/" + d[2]) as image:
        resized = T.functional.pad(image, padding=(14, 14, 15, 15), fill=256)
        if not os.path.exists("data/PACS_train/" + d[0] + "/" + d[1]):
            os.makedirs("data/PACS_train/" + d[0] + "/" + d[1])
        # resized.save("data/PACS_train/" + d[0] + "/" + d[1] + "/" + d[2])
        image.save("data/PACS_train/" + d[0] + "/" + d[1] + "/" + d[2])
    with open(f"data/PACS_train/meta/{d[0]}.txt", "a") as f:
        f.write(d[1] + "/" + d[2] + "\n")

if not os.path.exists("data/PACS_test/meta"):
    os.makedirs("data/PACS_test/meta")
for d in test_data:
    with Image.open("data/PACS/" + d[0] + "/" + d[1] + "/" + d[2]) as image:
        resized = T.functional.pad(image, padding=(14, 14, 15, 15), fill=256)
        if not os.path.exists("data/PACS_test/" + d[0] + "/" + d[1]):
            os.makedirs("data/PACS_test/" + d[0] + "/" + d[1])
        # resized.save("data/PACS_test/" + d[0] + "/" + d[1] + "/" + d[2])
        image.save("data/PACS_test/" + d[0] + "/" + d[1] + "/" + d[2])
    with open(f"data/PACS_test/meta/{d[0]}.txt", "a") as f:
        f.write(d[1] + "/" + d[2] + "\n")
