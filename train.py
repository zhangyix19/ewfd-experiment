import os
import time
from os.path import join

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

import attack as wfpattack
from argparser import parse_taskname, trainparser
from dataset import TraceDataset

# prase arguments
args = trainparser().parse_args()

random_seed = 11
log_root = "./run"
ds_root = "./data"
dump_root = "./data/dump"
time_str = time.strftime("%Y%m%d", time.localtime())
taskname = parse_taskname(args)
log_dir = join(log_root, time_str + args.note, args.attack, taskname)
dump_dir = join(dump_root, time_str + args.note, args.attack, taskname)
writer = SummaryWriter(log_dir)
writer.add_text("args", str(args))


# dataset
def get_dataset(dataset, defenses):
    global ds_root
    ds_dict = {}
    for defense in defenses:
        ds = TraceDataset(dataset, ds_root)
        ds.load_defended_by_name(defense, args.length)
        ds_dict[defense] = ds
    return ds_dict


total_defenses = args.train + args.test
ds_dict = get_dataset(args.dataset, total_defenses)
num_calsses = ds_dict[total_defenses[0]].num_classes()
ds_len = len(ds_dict[total_defenses[0]])


# preprocess
attack: wfpattack.DNNAttack = wfpattack.get_attack(args.attack)(args.length, num_calsses, args.gpu)
ds_dict = {name: attack.data_preprocess(*ds[:]) for name, ds in ds_dict.items()}

train_slice, valid_slice = train_test_split(
    [i for i in range(ds_len)], test_size=0.2, random_state=random_seed
)
train_features = torch.concat([ds_dict[defense]["traces"][train_slice] for defense in args.train])
train_labels = torch.concat([ds_dict[defense]["labels"][train_slice] for defense in args.train])
valid_features = torch.concat([ds_dict[defense]["traces"][valid_slice] for defense in args.train])
valid_labels = torch.concat([ds_dict[defense]["labels"][valid_slice] for defense in args.train])
test_data = {
    defense: {
        "features": ds_dict[defense]["traces"][valid_slice],
        "labels": ds_dict[defense]["labels"][valid_slice],
    }
    for defense in args.test
}
if args.dump or True:
    data_dump_dir = join(dump_dir, "data")
    os.makedirs(data_dump_dir, exist_ok=True)
    np.savez_compressed(
        join(data_dump_dir, "slice.npz"),
        train_slice=train_slice,
        valid_slice=valid_slice,
    )
    for defense in ds_dict:
        np.savez_compressed(
            join(data_dump_dir, defense + ".npz"),
            traces=ds_dict[defense]["traces"],
            labels=ds_dict[defense]["labels"],
        )
attack.init_model()
attack.train(
    train_features,
    train_labels,
    valid_features,
    valid_labels,
    writer=writer,
    save_root=dump_dir,
    test=test_data,
    num_epochs=args.epoch,
)
