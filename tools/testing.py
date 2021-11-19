import numpy as np
import pytorch_lightning as pl
# from pl_bolts.models.autoencoders import VAE
import torchvision.models as models

m = models.resnet101(pretrained=True)
