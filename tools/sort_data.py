import os
from PIL import Image
from tqdm import tqdm
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from wilds.common.grouper import CombinatorialGrouper

# Hyperparameters
dataset = "fmow"
domains = ["year", "region"]
datashape = (224, 224, -1)

# 
def get_groupname(dataset):
    if dataset == "poverty":
        return f"{int(metadata[0, 0].item())}_{int(metadata[0, 2].item())}"
    elif dataset == "fmow":
        return f"{int(metadata[0, 0].item())}_{int(metadata[0, 2].item())}"

# Main function
dataset = get_dataset(dataset=dataset, download=False)
train_data = dataset.get_subset("train")
train_loader = get_train_loader("standard", train_data, batch_size=1)
grouper = CombinatorialGrouper(dataset, domains)

highest = {}

for x, y_true, metadata in tqdm(train_loader):
    groupname = get_groupname(dataset)

    if groupname in list(highest.keys()):
        name = highest[groupname] + 1
    else:
        name = 0
    highest[groupname] = name

    if not os.path.exists(f"data/{dataset}_sorted/"):
        os.mkdir(f"data/{dataset}_sorted/")
    if not os.path.exists(f"data/{dataset}_sorted/{groupname}/"):
        os.mkdir(f"data/{dataset}_sorted/{groupname}/")
    
    im = Image.fromarray(x.numpy().reshape(datashape))
    im.save(f"{name}.jpeg")
