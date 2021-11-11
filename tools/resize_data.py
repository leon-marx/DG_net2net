import torch
import torchvision.transforms as T
import os
from PIL import Image

loadTrans = T.ToTensor()
saveTrans = T.ToPILImage()

for dir in os.listdir("data/PACS"):
    print(dir)
    for im in os.listdir("data/PACS/" + dir + "/dog"):
        print("   " + im)
        with Image.open("data/PACS/" + dir + "/dog/" + im) as image:
            resized = T.functional.pad(image, (14, 14, 15, 15))
            resized.save("data/PACS/" + dir + "/dog/" + im)