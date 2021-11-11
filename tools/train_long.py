import os
import subprocess

if os.name == "nt":
    os.chdir("C:/Users/gooog/Desktop/Bachelor/Code/bachelor")
else:
    os.chdir("/home/tarkus/leon/bachelor")

def create_yaml(filename, lr_expo, log_dir):
    with open(f"net2net/configs/creativity/{filename}", "w") as f1:
        with open("net2net/configs/creativity/pacs_dog_template.yaml", "r") as f2:
            for line in f2.readlines():
                line.strip()
                if "base_learning_rate:" in line.split():
                    f1.write(line[:-1] + str(lr_expo) + "\n")
                elif "ckpt_path:" in line.split():
                    f1.write(line[:-2] + log_dir + "\"\n")
                else:
                    f1.write(line)


for i in range(5):
    print("")
    print("")
    print("")
    print(f"Starting Iteration {i}")
    create_yaml("test", 10, "2021-22-11-T31_iteration_1")
    log_dir = ""
    if f"pacs_dog_{i}.yaml" not in os.listdir("net2net/configs/creativity"):
        lr_expo = 7+i
        log_dir = ""
        for j, name in enumerate(os.listdir("logs")):
            if f"iteration_{i-1}" in name:
                log_dir = os.listdir("logs")[j]
        log_dir += "/checkpoints/last.ckpt"
        create_yaml(f"pacs_dog_{i}.yaml", lr_expo, log_dir)
    print(f"Loading Model from {log_dir}")
    print("")
    bashCommand = f"python net2net/translation.py --base net2net/configs/creativity/pacs_dog_{i}.yaml -t --gpus 0,1,2,3 --name iteration_{i} --max_epochs 200"
    killCommand = "nvidia-smi | grep 'leon-net2net' | awk '{ print $5 }' | xargs -n1 kill -9"
    trainProcess = subprocess.Popen(bashCommand.split())
    trainOutput, trainError = trainProcess.communicate()
    trainProcess.wait()
    killProcess = subprocess.Popen(killCommand.split())
    killOutput, killError = killProcess.communicate()
    killProcess.wait()
