import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

for file in os.listdir("C:/users/gooog/desktop/bachelor/code/bachelor/logs/generated_net2net_2/"):
    plt.figure(figsize=(16, 16))
    plt.imshow(mpimg.imread("C:/users/gooog/desktop/bachelor/code/bachelor/logs/generated_net2net_2/" + file))
    plt.show()