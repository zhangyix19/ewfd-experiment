from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import pandas as pd
import sys
from os.path import join
import re
import dataset


def get_accuracy(log_file):
    print(f"{log_file=}")
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    paras_obj = ea.scalars.Items("valid/accuracy")
    return paras_obj[-1].value


def get_overhead(name, defense):
    ds = dataset.TraceDataset(name, cell_size=543 if name == "df" else 536)
    return ds.summary_overhead(defense)


note = sys.argv[1]
run_dir = join("run", note)
print(f"{run_dir=}")
for attack in os.listdir(run_dir):
    attack_dir = join(run_dir, attack)
    print(f"{attack_dir=}")
    # with open(f"log/{note}_{attack}.csv", "w") as f:
    data = pd.DataFrame()
    for name in os.listdir(attack_dir):
        assert (
            result := re.match(r"train_(?P<ds_name>\S+)_d(?P<defense>\S+)", name)
        ), f"Invalid task name: {name}"
        ds_name = result.groupdict()["ds_name"]
        defense = result.groupdict()["defense"]

        task_dir = join(attack_dir, name)
        log_file = os.listdir(task_dir)[0]
        print(f"{task_dir=}, {log_file=}")
        accuracy = get_accuracy(join(task_dir, log_file))
        overhead = get_overhead(ds_name, defense) if "&" not in defense else (-1, -1)
        data = pd.concat(
            [
                data,
                pd.Series(
                    [
                        ds_name,
                        defense,
                        round(accuracy, 4),
                        round(overhead[0], 4),
                        round(overhead[1], 4),
                    ]
                ),
            ],
            axis=1,
        )
    data.T.to_csv(
        f"log/{note}_{attack}.csv",
        index=False,
        header=["ds_name", "defense", "accuracy", "overhead_time", "overhead_data"],
    )
