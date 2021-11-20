import os
import subprocess

os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor")

bashCommand = "python net2net/translation.py --base net2net/configs/autoencoder/pacs_128.yaml -t --gpus 0, --name trash"
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
