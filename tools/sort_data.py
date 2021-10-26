import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper

dataset = get_dataset(dataset="poverty", download=False)
train_data = dataset.get_subset("train")
train_loader = get_train_loader("standard", train_data, batch_size=1)
grouper = CombinatorialGrouper(dataset, ["country", "urban"])

highest = {}

for x, y_true, metadata in tqdm(train_loader):
    # z = grouper.metadata_to_group(metadata).item()
    groupname = f"{int(metadata[0, 0].item())}_{int(metadata[0, 2].item())}"

    if groupname in list(highest.keys()):
        name = highest[groupname] + 1
    else:
        name = 0
    highest[groupname] = name

    if os.path.exists(f"data/poverty_sorted/{groupname}/"):
        im = Image.open(transforms.ToPILImage(x))
        im.save(f"{name}.jpeg")
    else:
        os.mkdir(f"data/poverty_sorted/{groupname}/")
        im = Image.open(transforms.ToPILImage(x))
        im.save(f"{name}.jpeg")
