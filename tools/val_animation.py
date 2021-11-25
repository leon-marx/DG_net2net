import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from PIL import Image

# Specify the Images and Figure size
images = []
for img in os.listdir("logs/2021-11-24T12-07-53_z_512_continued_3/val"):
    if "reconstruction" in img:
        image = np.asarray(Image.open("logs/2021-11-24T12-07-53_z_512_continued_3/val/" + img))
        images.append(image)

FIGSIZE = (18, 6)

# Animation is shown here
fig = plt.figure(figsize=FIGSIZE)
ax = plt.axes()
img = ax.imshow(np.zeros(images[0].shape))

def init():
    img.set_data(np.zeros(images[0].shape))
    return img,

def animate(i):
    img.set_data(images[i])
    return img,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(images), interval=1, blit=True)
plt.show()


