import os
import subprocess

# Hyperparameters
download_models = True
datadir = "data/fmow_sorted/"

def get_command(contentdir, styledir):
    return [f"th test.lua -contentDir {datadir + contentdir} -styleDir {datadir + styledir}"]

# Main function
if download_models:
    subprocess.run("bash models/download_models.sh")

for contentdir in os.listdir(datadir):
    for styledir in os.listdir(datadir):
        if contentdir != styledir:
            command = get_command(contentdir, styledir)

