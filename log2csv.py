from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import pandas as pd
import sys
from os.path import join
import re
import dataset


def get_accuracy(log_file):
    ea = event_accumulator.EventAccumulator(log_file)
    paras_obj = ea.scalars.Items("valid/accuracy")
    return paras_obj[-1].value


def get_overhead(dataset, defense):
    ds = dataset.TraceDataset(dataset)
    return ds.summary_overhead(defense)


note = sys.argv[1]
run_dir = join("run")
for attack in os.listdir(run_dir):
    task_dir = join(run_dir, attack)
    task_names = os.listdir(task_dir)
    # with open(f"log/{note}_{attack}.csv", "w") as f:
    data = pd.DataFrame()
    for name in os.listdir(task_dir):

        assert (result := re.match(r"train_(?P<dataset>\S+)_d(?P<defense>\S+)", name))
        dataset = result.groupdict()["dataset"]
        defense = result.groupdict()["defense"]

        task_dir = join(run_dir, name)
        log_file = os.listdir(task_dir)[0]
        accuracy = get_accuracy(join(task_dir, log_file))
        overhead = get_overhead(dataset, defense)
        data = pd.concat([data, pd.Series([dataset, defense, accuracy] + overhead)], axis=1)
    data.T.to_csv(f"log/{note}_{attack}.csv", index=False)
