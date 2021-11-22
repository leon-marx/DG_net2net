import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pl_bolts.models.self_supervised import resnets
from PIL import Image


# m = resnets.resnet101(pretrained=True)


img = Image.open("logs/2021-11-19T16-05-20_augmented_z_512/images/images/train/inputs_gs-000000_e-000000_b-000000.png")  # batch_size, channels, height, width
img = T.ToTensor()(img).view(1, 3, 260, -1)
print(img.shape)
print(img.max(), img.min())
plot1 = img.view(3, 260, -1).transpose(0, 2).transpose(0, 1).detach().numpy()
plt.imshow(plot1.astype(int) * 255)
plt.show()




"""
def get_x_prime(model):
    res = model.forward(img)
    return res







x_prime = get_x_prime(m)

print(x_prime.max(), x_prime.min())
x_prime += x_prime.min().abs()
x_prime /= x_prime.max().abs()
print(x_prime.max(), x_prime.min())

plot1 = img.view(3, 227, 227).transpose(0, 2).transpose(0, 1).detach().numpy()
plot2 = x_prime.view(3, 32, 32).transpose(0, 2).transpose(0, 1).detach().numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(plot1)
plt.subplot(1, 2, 2)
plt.title("Recovered")
plt.imshow(plot2)
plt.show()

m = VAE(input_height=227).load_from_checkpoint("logs/downloaded/cifar10_resnet18_epoch=89.ckpt")

# print(m.pretrained_weights_available())



img = Image.open("data/PACS_train/photo/dog/056_0050.jpg")  # batch_size, channels, height, width
img = T.ToTensor()(img).view(1, 3, 227, 227)
print(img.shape)
print(img.max(), img.min())

# Workflow of the VAE
x_prime = m.forward(img)
# mu = m.fc_mu(z_wiggle)
# log_var = m.fc_var(z_wiggle)
# p, q, z = m.sample(mu, log_var)
# x_prime = m.decoder(z)
# print(img.shape, img)
# print(z_wiggle.shape, z_wiggle)
# print(mu.shape, mu)
# print(log_var.shape, log_var)
# print(z.shape, z)
# print(x_prime.shape, x_prime)


print(x_prime.max(), x_prime.min())
x_prime += x_prime.min().abs()
x_prime /= x_prime.max().abs()
print(x_prime.max(), x_prime.min())

plot1 = img.view(3, 227, 227).transpose(0, 2).transpose(0, 1).detach().numpy()
plot2 = x_prime.view(3, 32, 32).transpose(0, 2).transpose(0, 1).detach().numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(plot1)
plt.subplot(1, 2, 2)
plt.title("Recovered")
plt.imshow(plot2)
plt.show()
"""