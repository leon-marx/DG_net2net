from typing import KeysView
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_path = "logs/2021-11-11T20-42-57_pacs_dog/testtube/version_0/metrics.csv"

df = pd.read_csv(log_path)
# print(df.head())
# print(df.tail())

print(list(df.keys()))

keys = list(df.keys())
keys = ["epoch_aeloss"]

print(df["created_at"])

for i, k in enumerate(keys):
    plt.figure(figsize=(12, 8))
    #plt.subplot(4, np.ceil(len(keys)/4), i+1)
    plt.scatter(df.index, df[k], label=k)
    plt.legend()
    plt.show()