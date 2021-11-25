import pandas as pd

content_list = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]

for content in content_list:
    df = pd.read_csv("tools/imagenet_classes/" + content + ".csv",
    names=["id", "name", "urls", "flickr_urls"])
    ids = df["id"][:50].to_numpy()
    # print(ids)
    class_list = ""
    for id in ids:
        class_list += id + " "
    # print(class_list)
    command = f"python ../imagenet-datasets-downloader/downloader.py -data_root /home/../hd-data/pacs_classes_only/{content} -use_class_list True -class_list {class_list}-images_per_class 10000"
    print(command)