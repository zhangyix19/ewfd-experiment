import os

from sklearn.model_selection import train_test_split

import attack as wfpattack
from argparser import evaluate_parser
from dataset import TraceDataset

# prase arguments
args = evaluate_parser().parse_args()
random_seed = 11
ds_root = "./data"
attack_name = args.attack
note = args.note
assert note.endswith("ow")
train = args.train
test = args.test
dataset = args.dataset
model_epoch = args.epoch
model_dir = (
    f"/data/users/zhangyixiang/wfp_zyx_new/data/dump/{note}/{attack_name}/train_{dataset}_d{train}"
)
assert os.path.exists(model_dir)

print("Loading raw dataset...")
raw_ds = TraceDataset(dataset, ds_root, scenario="open-world")
raw_ds.load_cell_level()
num_classes = raw_ds.num_classes()
attack: wfpattack.DNNAttack = wfpattack.get_attack(attack_name)(args.length, num_classes, args.gpu)

print("Loading test dataset...")
test_ds = TraceDataset(dataset, ds_root, scenario="open-world")
test_ds.load_defended_by_name(test)

ds_len = len(raw_ds)

_, evaluate_slice = train_test_split(
    [i for i in range(ds_len)], test_size=0.2, random_state=random_seed
)
print("Preparing test data...")
test_data = attack.data_preprocess(*test_ds[evaluate_slice])
test_features, test_labels = test_data["traces"], test_data["labels"]

attack.init_model()
print("Evaluating...")
result = attack.evaluate(
    test_features,
    test_labels,
    load_dir=model_dir,
    epoch=model_epoch,
    data=True,
)
assert isinstance(result, tuple) and len(result) == 3
metrics_dict, y_true, y_pred = result

import pickle

archive_dump = os.path.join(model_dir, "test")
os.makedirs(archive_dump, exist_ok=True)
with open(os.path.join(archive_dump, f"{test}.log"), "w") as f:
    f.write(str(metrics_dict))
pickle.dump(
    {"y_true": y_true, "y_pred": y_pred}, open(os.path.join(archive_dump, f"{test}.pkl"), "wb")
)

run_dump = os.path.join("run", "test", note, dataset, attack_name, train)
os.makedirs(run_dump, exist_ok=True)
with open(os.path.join(run_dump, f"{test}.log"), "w") as f:
    f.write(str(metrics_dict))
pickle.dump({"y_true": y_true, "y_pred": y_pred}, open(os.path.join(run_dump, f"{test}.pkl"), "wb"))
