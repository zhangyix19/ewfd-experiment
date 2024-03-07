# %%
import subprocess
from time import sleep

import numpy as np
import pynvml
from tqdm import tqdm

pynvml.nvmlInit()


def get_free_MB(i):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024 / 1024


def get_avail_gpu(size=10000):
    num = pynvml.nvmlDeviceGetCount()
    free = []
    for _ in range(60):
        sleep(1)
        free.append([get_free_MB(i) for i in range(num)])
    free = np.array(free).T.tolist()
    avail_gpus = []
    for i in range(num):
        if i == 3:
            continue
        if min(free[i]) > size:
            avail_gpus.append(i)
    return avail_gpus


# %%
tasks = []

for gan in [2, 4, 6, 8, 10]:
    for tol in [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]:
        tasks.append((f"switch_gan{gan:02d}_tol{tol}_front{10-gan:02d}", "switch_gan_front"))
for p in ["equal", "random"]:
    for tamaraw in range(2):
        for regulartor in range(2):
            for front in range(2):
                for wfgan in range(2):
                    if tamaraw + regulartor + front + wfgan > 0:
                        name = f"switch_{p}"
                        if front:
                            name += "_front"
                        if wfgan:
                            name += "_wfgan"
                        if tamaraw:
                            name += "_tamaraw"
                        if regulartor:
                            name += "_regulartor"
                        tasks.append((name, f"switch_{p}"))

# %%
avail_gpus = []
for task, note in tqdm(tasks):
    if not avail_gpus:
        while not (avail_gpus := get_avail_gpu()):
            pass
    else:
        gpu = avail_gpus.pop()
        subprocess.Popen(f"python train.py --train {task} --note {note} -g {gpu}", shell=True)
