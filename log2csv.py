from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import pandas as pd
import sys
from os.path import join


run_dir = join("run", sys.argv[1], sys.argv[2])
task_names = [d for d in os.listdir(run_dir) if os.path.isdir(join(run_dir, d))]
with open(f"csv/{sys.argv[1]}.csv", "w") as f:
    for task_name in task_names:
        task_dir = join(run_dir, task_name)
        task = os.listdir(task_dir)[0]
        path = join(task_dir, task)
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        paras_obj = ea.scalars.Items("valid/accuracy")
        print(",".join([task_name, str(paras_obj[-1].value)]), file=f)
