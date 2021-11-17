import numpy as np

t = [
    "elephant/n02503517_194-4.png",
    "house/8867.png",
    "dog/n02103406_343-9.png",
    "person/pic_111.jpg"
]

t2 = np.array(t, dtype=str)

print(t2)

def split(str):
    str = str.split("/")[0]
    return str

split_arr = np.vectorize(split)

plt_arr2 = np.vectorize(lambda s: s.split("/")[0])(t2)
print(plt_arr2)

print(split_arr(t2))